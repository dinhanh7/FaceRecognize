import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import torch
import torchvision
import os

yolo_trt_logger = trt.Logger(trt.Logger.INFO)
resnet_trt_logger = trt.Logger(trt.Logger.INFO)

def backbone_preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = img / 255.0
    img = (img - 0.5) / 0.5
    return np.ascontiguousarray(img[np.newaxis, ...], dtype=np.float32)

def yolo_preprocess(img, size=(640, 640)):
    h, w = img.shape[:2]
    img_input = cv2.resize(img, size)
    img_input = img_input[:, :, ::-1]      # BGR → RGB
    img_input = img_input.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, 0).astype(np.float32) / 255.0
    return img_input, (h, w)

def class_name(names):
    classes=[]
    file= open(names,'r')
    while True:
        name=file.readline().strip('\n')
        classes.append(name)
        if not name:
            break
    return classes

def xywh2xyxy(box):
    x, y, w, h = box
    return [
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    ]

def non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45):
    boxes = []
    for p in preds:
        obj_conf = p[4]
        if obj_conf < conf_thres:
            continue

        # YOLOv5 face: first 5 values = x,y,w,h,conf
        xyxy = xywh2xyxy(p[:4])
        boxes.append([*xyxy, obj_conf])

    # Convert to numpy
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return []

    # NMS
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    idxs = scores.argsort()[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area = (x2[idxs[1:]] - x1[idxs[1:]]) * (y2[idxs[1:]] - y1[idxs[1:]])
        iou = inter / (area_i + area - inter)

        idxs = idxs[1:][iou < iou_thres]

    return boxes[keep]

def load_database(db_dir="database_tensor"):
    names = []
    feats = []
    for f in os.listdir(db_dir):
        if f.endswith(".npy"):
            names.append(f.replace(".npy", ""))
            feats.append(np.load(os.path.join(db_dir, f)))
    feats = np.vstack(feats)  # (N, feat_dim)
    return names, feats

def match_face(feat, db_feats, db_names, thres=0.3):
    def l2_normalize(x, axis=1, eps=1e-10):
        return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)
    # Normalize
    feat = l2_normalize(feat[np.newaxis, :])[0]   # (D,)
    db_feats = l2_normalize(db_feats)              # (N, D)

    # Cosine similarity = dot product (sau normalize)
    sims = np.dot(db_feats, feat)                  # (N,)

    idx = np.argmax(sims)
    best_sim = sims[idx]

    if best_sim >= thres:
        return db_names[idx], best_sim
    else:
        return "unknown", best_sim


class trt_v5:
    def __init__(self, vid_path, yolo_trt, resnet_trt,
                 img_size, classes, webcam=False,
                 conf_thres=0.25, iou_thres=0.45):

        self.webcam = webcam
        self.vid_path = vid_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.class_file = classes

        # -------- Load YOLO engine --------
        with open(yolo_trt, "rb") as f:
            runtime = trt.Runtime(yolo_trt_logger)
            self.yolo_engine = runtime.deserialize_cuda_engine(f.read())
        self.yolo_context = self.yolo_engine.create_execution_context()

        # -------- Load ResNet engine --------
        with open(resnet_trt, "rb") as f:
            runtime = trt.Runtime(resnet_trt_logger)
            self.resnet_engine = runtime.deserialize_cuda_engine(f.read())
        self.resnet_context = self.resnet_engine.create_execution_context()

        # -------- Load database --------
        self.db_names, self.db_feats = load_database()

        self.stream = cuda.Stream()

        self._alloc_yolo()
        self._alloc_resnet()

    def __call__(self):
        cap = cv2.VideoCapture(0 if self.webcam else self.vid_path)

        writer = None
        log = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.detect_face(frame)

            for x1,y1,x2,y2,name,conf in results:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,name,(x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                log.append(name)

            if writer is None and not self.webcam:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter("output.mp4", fourcc, 25,
                                         (frame.shape[1], frame.shape[0]))

            if writer:
                writer.write(frame)

            cv2.imshow("Face TRT", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        with open("recognition_log.txt","w") as f:
            for n in log:
                f.write(n + "\n")

    def _alloc_yolo(self):
        for i in range(self.yolo_engine.num_io_tensors):
            name = self.yolo_engine.get_tensor_name(i)
            if self.yolo_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.yolo_input = name
            else:
                self.yolo_output = name

        self.yolo_in_shape = self.yolo_engine.get_tensor_shape(self.yolo_input)
        self.yolo_out_shape = self.yolo_engine.get_tensor_shape(self.yolo_output)

        self.h_yolo_in = np.empty(self.yolo_in_shape, np.float32)
        self.h_yolo_out = np.empty(self.yolo_out_shape, np.float32)

        self.d_yolo_in = cuda.mem_alloc(self.h_yolo_in.nbytes)
        self.d_yolo_out = cuda.mem_alloc(self.h_yolo_out.nbytes)
   
    def _alloc_resnet(self):
        for i in range(self.resnet_engine.num_io_tensors):
            name = self.resnet_engine.get_tensor_name(i)
            if self.resnet_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.resnet_input = name
            else:
                self.resnet_output = name

        self.resnet_in_shape = self.resnet_engine.get_tensor_shape(self.resnet_input)
        self.resnet_out_shape = self.resnet_engine.get_tensor_shape(self.resnet_output)

        self.h_resnet_in = np.empty(self.resnet_in_shape, np.float32)
        self.h_resnet_out = np.empty(self.resnet_out_shape, np.float32)

        self.d_resnet_in = cuda.mem_alloc(self.h_resnet_in.nbytes)
        self.d_resnet_out = cuda.mem_alloc(self.h_resnet_out.nbytes)

    def detect_face(self, frame):
        img, (H, W) = yolo_preprocess(frame, self.img_size)
        self.h_yolo_in[:] = img

        cuda.memcpy_htod_async(self.d_yolo_in, self.h_yolo_in, self.stream)
        self.yolo_context.set_tensor_address(self.yolo_input, int(self.d_yolo_in))
        self.yolo_context.set_tensor_address(self.yolo_output, int(self.d_yolo_out))

        self.yolo_context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_yolo_out, self.d_yolo_out, self.stream)
        self.stream.synchronize()

        preds = self.h_yolo_out.reshape(-1, 16)
        boxes = non_max_suppression(preds, self.conf_thres, self.iou_thres)

        results = []
        for (x1, y1, x2, y2, conf) in boxes:
            x1 = int(x1 * W / 640)
            x2 = int(x2 * W / 640)
            y1 = int(y1 * H / 640)
            y2 = int(y2 * H / 640)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            feat = self.extract_feat(face)
            name, score = match_face(feat, self.db_feats, self.db_names)

            results.append((x1, y1, x2, y2, name, conf))

        return results
    
    def extract_feat(self, face):
        img = backbone_preprocess(face)
        self.h_resnet_in[:] = img

        cuda.memcpy_htod_async(self.d_resnet_in, self.h_resnet_in, self.stream)
        self.resnet_context.set_tensor_address(self.resnet_input, int(self.d_resnet_in))
        self.resnet_context.set_tensor_address(self.resnet_output, int(self.d_resnet_out))

        self.resnet_context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_resnet_out, self.d_resnet_out, self.stream)
        self.stream.synchronize()

        return self.h_resnet_out.squeeze()


if __name__ == "__main__":
    yolo_trt_file = "./models/yolov5m-face_fp16.engine"
    resnet_trt_file = "./models/backbone_fp16.engine"
    classes_file = "./test/classes.txt"
    video_path = "video.mp4"

    # Khởi tạo pipeline
    detector = trt_v5(
        vid_path=video_path,
        yolo_trt=yolo_trt_file,
        resnet_trt=resnet_trt_file,
        img_size=(640, 640),
        classes=classes_file,
        webcam=True,          
        conf_thres=0.25,
        iou_thres=0.45
    )

    # Chạy inference
    detector()
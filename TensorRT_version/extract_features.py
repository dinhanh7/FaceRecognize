import argparse
import cv2
import numpy as np
import os
from pathlib import Path

from models_init import ExtractTRT  

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = img / 255.0
    img = (img - 0.5) / 0.5
    return np.ascontiguousarray(img[np.newaxis, ...], dtype=np.float32)   # (1,3,112,112)

def inference_trt(engine, img):
    inp = preprocess(img)
    feat = engine.infer(inp)
    return feat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TensorRT ResNet Inference')
    parser.add_argument('--engine', type=str, default = "./models/backbone_fp16.engine", help='path to resnet.engine')
    parser.add_argument('--path_database', type=Path, default="database_image")
    args = parser.parse_args()

    # Load TensorRT model instead of PyTorch
    engine = ExtractTRT(args.engine)

    # create output folder
    if not os.path.exists('database_tensor'):
        os.mkdir('database_tensor')

    path = args.path_database
    imgs = os.listdir(path)

    for im in imgs:
        full = str(path / im)
        feat = inference_trt(engine, full)

        np.save('database_tensor/' + im.replace('.jpg', '') + '.npy', feat)
        print("Saved:", im)

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class ExtractTRT:
    def __init__(self, engine_path):
        engine_path = engine_path.strip()

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # ----- LẤY TÊN INPUT/OUTPUT -----
        self.input_name = None
        self.output_name = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        print("INPUT =", self.input_name)
        print("OUTPUT =", self.output_name)

        # LẤY SHAPE
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        # HOST + DEVICE buffers
        self.h_input = np.empty(self.input_shape, dtype=np.float32)
        self.h_output = np.empty(self.output_shape, dtype=np.float32)

        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        self.stream = cuda.Stream()

    def infer(self, input_np):
        # Ensure contiguous
        input_np = np.ascontiguousarray(input_np.astype(np.float32))

        # Copy input
        cuda.memcpy_htod(self.d_input, input_np)

        # Bind
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        self.context.execute_async_v3(self.stream.handle)

        # Copy result
        cuda.memcpy_dtoh(self.h_output, self.d_output)

        return self.h_output.copy()
    
# class YOLOv5TRT:
#     def __init__(self, engine_path):
#         self.logger = trt.Logger(trt.Logger.INFO)

#         # Load engine
#         with open(engine_path, "rb") as f:
#             runtime = trt.Runtime(self.logger)
#             self.engine = runtime.deserialize_cuda_engine(f.read())

#         self.context = self.engine.create_execution_context()

#         # Find input & output tensor names
#         self.input_name = None
#         self.output_name = None
#         for i in range(self.engine.num_io_tensors):
#             name = self.engine.get_tensor_name(i)
#             mode = self.engine.get_tensor_mode(name)

#             if mode == trt.TensorIOMode.INPUT:
#                 self.input_name = name
#             elif mode == trt.TensorIOMode.OUTPUT:
#                 self.output_name = name

#         print("[TRT] Input :", self.input_name)
#         print("[TRT] Output:", self.output_name)

#         # Allocate memory
#         self.input_shape = self.engine.get_tensor_shape(self.input_name)
#         self.output_shape = self.engine.get_tensor_shape(self.output_name)

#         self.h_input = np.zeros(self.input_shape, dtype=np.float32)
#         self.h_output = np.zeros(self.output_shape, dtype=np.float32)

#         self.d_input = cuda.mem_alloc(self.h_input.nbytes)
#         self.d_output = cuda.mem_alloc(self.h_output.nbytes)

#         self.stream = cuda.Stream()

#     # --------------------------------------------------
#     # Preprocess (Resize 640x640, Normalize)
#     # --------------------------------------------------
#     def preprocess(self, img):
#         img_resized = cv2.resize(img, (640, 640))
#         img_resized = img_resized[:, :, ::-1]   # BGR -> RGB
#         img_resized = img_resized.transpose(2, 0, 1)
#         img_resized = img_resized.astype(np.float32) / 255.0
#         return np.expand_dims(img_resized, 0)

#     # --------------------------------------------------
#     # Run inference
#     # --------------------------------------------------
#     def infer(self, img):
#         inp = self.preprocess(img)
#         self.h_input[:] = inp

#         # Copy to device
#         cuda.memcpy_htod(self.d_input, self.h_input)

#         # Bind
#         self.context.set_tensor_address(self.input_name, int(self.d_input))
#         self.context.set_tensor_address(self.output_name, int(self.d_output))

#         # Execute
#         self.context.execute_async_v3(self.stream.handle)
#         self.stream.synchronize()

#         # Copy back
#         cuda.memcpy_dtoh(self.h_output, self.d_output)

#         # YOLO output: (1,25200,16)
#         return self.h_output.reshape(-1, 16)

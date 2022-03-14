import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("/d/pfournie/tuto_depthai/test.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

outputs = ort_session.run(
        None,
        {"actual_input_1": np.random.randn(1, 3, 224, 224).astype(np.float32)}
        )
print(outputs[0])

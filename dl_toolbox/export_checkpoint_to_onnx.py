import numpy as np
import torch
from lightning_modules import SupervisedBaseline
import onnx
import onnxruntime


def main():

    # Loading the module used for training with the weights from the checkpoint.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
        '/d/pfournie/dl_toolbox/outputs/test_semcity/epoch=100-step=62500.ckpt', map_location=device)

    module = SupervisedBaseline(
        in_channels=3,
        num_classes=8
    )
    module.load_state_dict(ckpt['state_dict'])
    module.eval()
    module.network.encoder.set_swish(memory_efficient=False)
    module.to(device)

    dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True, device=device)
    dummy_output = module(dummy_input)

    torch.onnx.export(
        module, dummy_input, 'test_semcity.onnx', export_params=True, opset_version=11, input_names=['input'], output_names=['output'],
        # dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 'output': {0: 'batch_size', 2: 'height', 3: 'width'}},
        do_constant_folding=True
    )

    onnx_model = onnx.load("test_semcity.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("test_semcity.onnx", providers=[
                                              'CUDAExecutionProvider', 'CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(
        to_numpy(dummy_output), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':

    main()

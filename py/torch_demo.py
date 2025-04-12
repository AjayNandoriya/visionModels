import torch

def load_model():
    # Load a pre-trained SqueezeNet model from PyTorch's model hub
    # The model is loaded with pre-trained weights for ImageNet
    model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)

    # convert to onnx
    dummy_input = torch.randn(1, 3, 224, 224)  # Example input tensor
    onnx_file_path = "squeezenet.onnx"
    torch.onnx.export(model, dummy_input, onnx_file_path, export_params=True, opset_version=11)
    print(f"Model exported to {onnx_file_path}")
    return model

if __name__ == "__main__":
    load_model()
    print("Model loaded successfully.")
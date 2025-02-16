import torch

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("GPU is not available, using CPU")
        device = torch.device("cpu")

    # Perform a simple tensor operation to verify GPU usage
    x = torch.rand(3, 3, device=device)
    print(f"Tensor on device {device}:\n{x}")

if __name__ == "__main__":
    check_gpu()

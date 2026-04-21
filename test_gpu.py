import torch

def test_gpu():
    print("--- PyTorch GPU Test ---")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"Devices Found:   {device_count}")
        print(f"Active Device:   {device_name}")
        
        print("\nTesting VRAM Allocation...")
        try:
            # Create a dummy tensor and push it to the GPU
            test_tensor = torch.rand(5, 5).to('cuda')
            print("✅ SUCCESS: Tensor successfully allocated to the RTX 3050!")
        except Exception as e:
            print(f"❌ ERROR: Failed to allocate memory to GPU. Details: {e}")
    else:
        print("\n❌ FAIL: PyTorch cannot see your GPU. It will fall back to your CPU (which will be too slow).")

if __name__ == "__main__":
    test_gpu()
import torch
from src.model import create_model

def test_model():
    print("Testing SwinIRLight model initialization...")
    model = create_model(upscale=2)
    print(f"Model created with success.")
    
    # Test with dummy input
    # (B, C, H, W)
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (1, 3, 128, 128), "Output shape mismatch!"
    print("Test passed!")

if __name__ == "__main__":
    test_model()

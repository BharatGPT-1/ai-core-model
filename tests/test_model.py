import torch
from core.model import TransformerModel

def test_model_output_shapes():
    model = TransformerModel(vocab_size=1000, d_model=64)
    x = torch.randint(0, 1000, (2, 32))
    output = model(x)
    assert output.shape == (2, 32, 1000)
    
def test_gradient_flow():
    model = TransformerModel(vocab_size=1000, d_model=64)
    x = torch.randint(0, 1000, (2, 32))
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
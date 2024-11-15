import torch
import torch.nn as nn

# Setup
batch_size = 2
seq_len = 4
embed_dim = 32
num_heads = 8
head_dim = embed_dim // num_heads

# Input tensor
x = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float64)

# Approach 1: Single linear layer then split
single_linear = nn.Linear(embed_dim, embed_dim, dtype=torch.float64)
out1 = single_linear(x)  # [batch, seq_len, embed_dim]
# Reshape to [batch, seq_len, num_heads, head_dim]
out1_reshaped = out1.view(batch_size, seq_len, num_heads, head_dim)


# Approach 2: Separate linear layer per head
class MultipleLinears(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_dim, head_dim, dtype=torch.float64)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        # Process each head separately
        head_outputs = [linear(x) for linear in self.linears]
        # Stack along new head dimension
        return torch.stack(head_outputs, dim=2)


multi_linear = MultipleLinears()

# Process input
out2 = multi_linear(x)  # [batch, seq_len, num_heads, head_dim]


# Now we can show how to convert between the approaches
def convert_single_to_multi(single_layer):
    """Convert weights from single linear to per-head linears"""
    weight = single_layer.weight.view(num_heads, head_dim, embed_dim)
    bias = (
        single_layer.bias.view(num_heads, head_dim)
        if single_layer.bias is not None
        else None
    )

    multi = MultipleLinears()
    for h in range(num_heads):
        multi.linears[h].weight.data = weight[h]
        if bias is not None:
            multi.linears[h].bias.data = bias[h]

    return multi


# Demo: We can achieve the same output either way
def test_equivalence():
    # Initialize a single linear layer
    single = nn.Linear(embed_dim, embed_dim, dtype=torch.float64)

    # Convert it to multiple heads
    multi = convert_single_to_multi(single)

    # Process same input
    x = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float64)

    out_single = single(x).view(batch_size, seq_len, num_heads, head_dim)
    out_multi = multi(x)

    # Should be very close (allowing for floating point differences)
    print(f"Max difference: {(out_single - out_multi).abs().max().item():.64f}")
    pass


test_equivalence()

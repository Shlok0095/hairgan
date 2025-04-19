import os
import sys
from pathlib import Path

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def write_cpu_implementation():
    # Path to the StyleGAN2 op directory
    op_dir = "HairFastGAN/models/stylegan2/op"
    ensure_dir(op_dir)
    
    # Write __init__.py with CPU implementations
    init_content = """import torch
from torch import nn
import torch.nn.functional as F

# CPU implementations

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.channel = channel
        self.negative_slope = negative_slope
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(channel))

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope) * scale

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    # Use pure PyTorch implementation
    out = _upfirdn2d_pytorch(
        input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )
    return out

def _upfirdn2d_pytorch(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    input = input.float()
    kernel = kernel.float()

    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out = out.view(-1, channel, out.shape[1], out.shape[2])

    return out
"""
    
    # Create dummy files for fused_act.py and upfirdn2d.py
    dummy_content = "# CPU backup file - Replaced with CPU implementation in __init__.py"
    
    with open(os.path.join(op_dir, "__init__.py"), "w") as f:
        f.write(init_content)
    
    with open(os.path.join(op_dir, "fused_act.py"), "w") as f:
        f.write(dummy_content)
    
    with open(os.path.join(op_dir, "upfirdn2d.py"), "w") as f:
        f.write(dummy_content)
    
    print("Successfully applied CPU implementation for StyleGAN2 operations.")

def modify_hair_model():
    # Add CPU mode enforcement to hair_model.py
    hair_model_path = "app/models/hair_model.py"
    with open(hair_model_path, "r") as f:
        content = f.read()
    
    # Add CPU mode code if it's not already there
    if "os.environ['CUDA_VISIBLE_DEVICES'] = '-1'" not in content:
        cpu_code = """
# Ensure CPU mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""
        lines = content.split("\n")
        # Find line after imports to insert our code
        for i, line in enumerate(lines):
            if "import" in line and i > 3:
                lines.insert(i+1, cpu_code)
                break
        
        with open(hair_model_path, "w") as f:
            f.write("\n".join(lines))
        
        print("Successfully modified hair_model.py to force CPU mode.")

if __name__ == "__main__":
    print("Fixing CUDA issues in HairFastGAN...")
    write_cpu_implementation()
    modify_hair_model()
    print("Done! You can now run the application with: python app.py") 
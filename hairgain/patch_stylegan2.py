import os
import sys
import torch
from pathlib import Path

def apply_patch():
    """Apply patches to StyleGAN2 op folder to support GTX 1650"""
    op_dir = "HairFastGAN/models/stylegan2/op"
    Path(op_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Applying CPU-only patch.")
        apply_cpu_patch(op_dir)
        return
    
    # Check if GPU is GTX 1650
    device_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {device_name}")
    
    if "GTX 1650" in device_name:
        print("Applying special patch for GTX 1650...")
        apply_gtx1650_patch(op_dir)
    else:
        print("Using default implementation.")

def apply_cpu_patch(op_dir):
    """Apply CPU-only patch"""
    # Create CPU implementation of operations
    cpu_code = """import torch
from torch import nn
import torch.nn.functional as F

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
    
    with open(os.path.join(op_dir, "__init__.py"), "w") as f:
        f.write(cpu_code)
    
    # Create dummy files to prevent import errors
    with open(os.path.join(op_dir, "fused_act.py"), "w") as f:
        f.write("# CPU backup file")
    
    with open(os.path.join(op_dir, "upfirdn2d.py"), "w") as f:
        f.write("# CPU backup file")
    
    print("Applied CPU-only patch.")

def apply_gtx1650_patch(op_dir):
    """Apply patch specific for GTX 1650"""
    # Create a patched fused_act.py
    fused_act_code = """
import os

import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

# Custom version for GTX 1650
class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, input, bias, negative_slope, scale):
        ctx.save_for_backward(grad_output, input, bias)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return grad_output

    @staticmethod
    def backward(ctx, gradgrad_output):
        grad_output, input, bias = ctx.saved_tensors
        negative_slope = ctx.negative_slope
        scale = ctx.scale

        return gradgrad_output, None, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        # GTX 1650 friendly implementation
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        result = F.leaky_relu(
            input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
        ) * scale

        ctx.save_for_backward(input, bias)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        negative_slope = ctx.negative_slope
        scale = ctx.scale

        # Modified for GTX 1650
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        input_p_bias = input + bias.view(1, bias.shape[0], *rest_dim)
        grad_input = torch.where(
            input_p_bias > 0,
            grad_output * scale,
            grad_output * scale * negative_slope
        )

        grad_bias = torch.sum(
            torch.where(
                input_p_bias > 0,
                grad_output * scale,
                grad_output * scale * negative_slope
            ),
            dim=[0] + list(range(2, grad_output.ndim))
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    # Modified for GTX 1650
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
"""
    
    # Create a patched init file that uses modified fused_act.py but PyTorch upfirdn2d
    init_code = """from .fused_act import FusedLeakyReLU, fused_leaky_relu

import torch
from torch.nn import functional as F

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    # Use pure PyTorch implementation for GTX 1650
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
    
    # Write the patched files
    with open(os.path.join(op_dir, "fused_act.py"), "w") as f:
        f.write(fused_act_code)
    
    with open(os.path.join(op_dir, "__init__.py"), "w") as f:
        f.write(init_code)
    
    # Create upfirdn2d file (unused but prevents import errors)
    with open(os.path.join(op_dir, "upfirdn2d.py"), "w") as f:
        f.write("# Patched version for GTX 1650")
    
    print("Applied GTX 1650 specific patch.")

if __name__ == "__main__":
    print("Patching StyleGAN2 to work with your GPU...")
    apply_patch()
    print("Patch completed. You can now run: python app.py") 
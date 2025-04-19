@echo off
echo Fixing CUDA issues in HairFastGAN...

REM Create directory if it doesn't exist
mkdir HairFastGAN\models\stylegan2\op 2>nul

REM Create a simplified __init__.py that works without CUDA
echo import torch
echo from torch import nn
echo import torch.nn.functional as F
echo.
echo # CPU implementations
echo.
echo class FusedLeakyReLU(nn.Module):
echo     def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
echo         super().__init__()
echo         self.channel = channel
echo         self.negative_slope = negative_slope
echo         self.scale = scale
echo         self.bias = nn.Parameter(torch.zeros(channel))
echo.
echo     def forward(self, input):
echo         return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
echo.
echo def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
echo     rest_dim = [1] * (input.ndim - bias.ndim - 1)
echo     return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope) * scale
echo.
echo def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
echo     # Use pure PyTorch implementation
echo     out = _upfirdn2d_pytorch(
echo         input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
echo     )
echo     return out
echo.
echo def _upfirdn2d_pytorch(
echo     input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
echo ):
echo     input = input.float()
echo     kernel = kernel.float()
echo.
echo     _, channel, in_h, in_w = input.shape
echo     input = input.reshape(-1, in_h, in_w, 1)
echo.
echo     _, in_h, in_w, minor = input.shape
echo     kernel_h, kernel_w = kernel.shape
echo.
echo     out = input.view(-1, in_h, 1, in_w, 1, minor)
echo     out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
echo     out = out.view(-1, in_h * up_y, in_w * up_x, minor)
echo.
echo     out = F.pad(
echo         out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
echo     )
echo     out = out[
echo         :,
echo         max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
echo         max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
echo         :,
echo     ]
echo.
echo     out = out.permute(0, 3, 1, 2)
echo     out = out.reshape(
echo         [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
echo     )
echo     w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
echo     out = F.conv2d(out, w)
echo     out = out.reshape(
echo         -1,
echo         minor,
echo         in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
echo         in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
echo     )
echo     out = out.permute(0, 2, 3, 1)
echo     out = out[:, ::down_y, ::down_x, :]
echo.
echo     out = out.view(-1, channel, out.shape[1], out.shape[2])
echo.
echo     return out > HairFastGAN\models\stylegan2\op\__init__.py

REM Create dummy backup files to prevent import errors
echo # CPU backup file > HairFastGAN\models\stylegan2\op\fused_act.py
echo # CPU backup file > HairFastGAN\models\stylegan2\op\upfirdn2d.py

echo Done! The HairFastGAN should now work without CUDA.
echo You can now run: python app.py
pause 
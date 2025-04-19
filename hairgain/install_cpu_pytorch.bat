@echo off
echo Installing CPU-only PyTorch...

pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo PyTorch CPU version installed.
echo Now try running the application again with: python app.py
pause 
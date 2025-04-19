@echo off
echo Setting up HairFastGAN Web API for Windows (CPU mode)...

REM Create directories
echo Creating directories...
mkdir uploads 2>nul
mkdir results 2>nul
mkdir app\static\results 2>nul

REM Clone HairFastGAN repository if not exists
if not exist HairFastGAN (
    echo Cloning HairFastGAN repository...
    git clone https://github.com/AIRI-Institute/HairFastGAN
    if %ERRORLEVEL% neq 0 (
        echo Failed to clone HairFastGAN repository.
        goto :error
    )
)

REM Install CPU PyTorch
echo Installing CPU PyTorch...
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if %ERRORLEVEL% neq 0 (
    echo Failed to install CPU PyTorch.
    goto :error
)

REM Install compatible Flask versions
echo Installing Flask dependencies...
pip uninstall -y flask flask-restx Werkzeug
pip install flask==2.0.1 flask-restx==0.5.1 Werkzeug==2.0.1
if %ERRORLEVEL% neq 0 (
    echo Failed to install Flask dependencies.
    goto :error
)

REM Install HairFastGAN requirements
echo Installing HairFastGAN dependencies...
pip install -r HairFastGAN\requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install HairFastGAN dependencies.
    goto :error
)

REM Fix CUDA issues using Python script
echo Fixing CUDA issues...
python fix_cuda.py
if %ERRORLEVEL% neq 0 (
    echo Failed to fix CUDA issues.
    goto :error
)

REM Download pretrained models if not exists
if not exist pretrained_models (
    echo Downloading pretrained models...
    
    REM Check if git lfs is installed
    git lfs install
    if %ERRORLEVEL% neq 0 (
        echo Git LFS not installed. Please install Git LFS to download model files.
        echo Visit https://git-lfs.github.com/ for installation instructions.
        goto :error
    )
    
    git clone https://huggingface.co/AIRI-Institute/HairFastGAN HairFastGAN_models
    if %ERRORLEVEL% neq 0 (
        echo Failed to clone models repository.
        goto :error
    )
    
    cd HairFastGAN_models
    git lfs pull
    if %ERRORLEVEL% neq 0 (
        echo Failed to download model files with git-lfs.
        cd ..
        goto :error
    )
    cd ..
    
    echo Setting up model files...
    mkdir pretrained_models 2>nul
    mkdir input 2>nul
    
    if exist HairFastGAN_models\pretrained_models (
        xcopy /E /I /Y HairFastGAN_models\pretrained_models pretrained_models
    )
    
    if exist HairFastGAN_models\input (
        xcopy /E /I /Y HairFastGAN_models\input input
    )
    
    REM Cleanup
    echo Cleaning up temporary files...
    rmdir /S /Q HairFastGAN_models
)

echo.
echo Setup completed successfully! (CPU mode)
echo.
echo To run the application, use the command:
echo python app.py
echo.
echo The application will be available at:
echo - Web interface: http://localhost:5000
echo - API documentation: http://localhost:5000/api/doc
goto :end

:error
echo.
echo Setup failed. Please check the error messages above.

:end
pause 
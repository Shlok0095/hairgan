@echo off
echo Setting up HairFastGAN Web API...

REM Create directories
mkdir uploads 2>nul
mkdir results 2>nul

REM Clone HairFastGAN repository if not exists
if not exist HairFastGAN (
    echo Cloning HairFastGAN repository...
    git clone https://github.com/AIRI-Institute/HairFastGAN
    if %ERRORLEVEL% neq 0 (
        echo Failed to clone HairFastGAN repository.
        goto :error
    )
)

REM Install dependencies
echo Installing dependencies...
pip install -r HairFastGAN\requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install HairFastGAN dependencies.
    goto :error
)

pip install flask==2.2.3 flask-restx==1.1.0 Werkzeug==2.2.3
if %ERRORLEVEL% neq 0 (
    echo Failed to install Flask dependencies.
    goto :error
)

REM Download pretrained models if not exists
if not exist pretrained_models (
    echo Downloading pretrained models...
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
    if exist HairFastGAN_models\pretrained_models (
        xcopy /E /I /Y HairFastGAN_models\pretrained_models pretrained_models
    )
    
    if exist HairFastGAN_models\input (
        xcopy /E /I /Y HairFastGAN_models\input input
    )
    
    REM Cleanup
    rmdir /S /Q HairFastGAN_models
)

echo Setup completed successfully!
echo Run 'python app.py' to start the application.
goto :end

:error
echo Setup failed.

:end
pause 
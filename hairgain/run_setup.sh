#!/bin/bash

echo "Setting up HairFastGAN Web API..."

# Create directories
mkdir -p uploads
mkdir -p results

# Clone HairFastGAN repository if not exists
if [ ! -d "HairFastGAN" ]; then
    echo "Cloning HairFastGAN repository..."
    git clone https://github.com/AIRI-Institute/HairFastGAN
    if [ $? -ne 0 ]; then
        echo "Failed to clone HairFastGAN repository."
        exit 1
    fi
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r HairFastGAN/requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install HairFastGAN dependencies."
    exit 1
fi

pip install flask==2.2.3 flask-restx==1.1.0 Werkzeug==2.2.3
if [ $? -ne 0 ]; then
    echo "Failed to install Flask dependencies."
    exit 1
fi

# Download pretrained models if not exists
if [ ! -d "pretrained_models" ]; then
    echo "Downloading pretrained models..."
    git clone https://huggingface.co/AIRI-Institute/HairFastGAN HairFastGAN_models
    if [ $? -ne 0 ]; then
        echo "Failed to clone models repository."
        exit 1
    fi
    
    cd HairFastGAN_models
    git lfs pull
    if [ $? -ne 0 ]; then
        echo "Failed to download model files with git-lfs."
        cd ..
        exit 1
    fi
    cd ..
    
    echo "Setting up model files..."
    if [ -d "HairFastGAN_models/pretrained_models" ]; then
        mv HairFastGAN_models/pretrained_models .
    fi
    
    if [ -d "HairFastGAN_models/input" ]; then
        mv HairFastGAN_models/input .
    fi
    
    # Cleanup
    rm -rf HairFastGAN_models
fi

echo "Setup completed successfully!"
echo "Run 'python app.py' to start the application." 
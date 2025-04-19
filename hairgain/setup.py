#!/usr/bin/env python
"""Setup script for HairFastGAN Web API"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command):
    """Run a command with proper error handling"""
    try:
        subprocess.run(command, check=True, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

def setup():
    """Setup the HairFastGAN Web API"""
    # Create directories if they don't exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Clone HairFastGAN repository if not already cloned
    if not Path("HairFastGAN").exists():
        print("Cloning HairFastGAN repository...")
        if not run_command("git clone https://github.com/AIRI-Institute/HairFastGAN"):
            print("Failed to clone HairFastGAN repository. Please check your internet connection.")
            return False
    
    # Install dependencies
    print("Installing dependencies...")
    if not run_command("pip install -r HairFastGAN/requirements.txt"):
        print("Failed to install dependencies from HairFastGAN.")
        return False
    
    if not run_command("pip install flask==2.2.3 flask-restx==1.1.0 Werkzeug==2.2.3"):
        print("Failed to install Flask dependencies.")
        return False
    
    # Clone and setup pretrained models if they don't exist
    if not Path("pretrained_models").exists():
        print("Downloading pretrained models...")
        if not run_command("git clone https://huggingface.co/AIRI-Institute/HairFastGAN HairFastGAN_models"):
            print("Failed to clone models repository. Please check your internet connection.")
            return False
        
        if not run_command("cd HairFastGAN_models && git lfs pull"):
            print("Failed to download model files with git-lfs.")
            return False
        
        # Move model files
        print("Setting up model files...")
        if Path("HairFastGAN_models/pretrained_models").exists():
            shutil.move("HairFastGAN_models/pretrained_models", "pretrained_models")
        
        if Path("HairFastGAN_models/input").exists():
            shutil.move("HairFastGAN_models/input", "input")
        
        # Cleanup
        shutil.rmtree("HairFastGAN_models", ignore_errors=True)
    
    print("Setup completed successfully!")
    print("Run 'python app.py' to start the application.")
    return True

if __name__ == "__main__":
    setup() 
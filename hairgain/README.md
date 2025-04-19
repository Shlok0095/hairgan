# HairFastGAN Web API

A Flask-based web application with REST API for the HairFastGAN hair style generation model.

## Installation

### 1. Clone the repositories

First, clone this repository and the HairFastGAN repository:

```bash
git clone https://github.com/AIRI-Institute/HairFastGAN
```

### 2. Install dependencies

The HairFastGAN repository includes all the necessary dependencies. Install them with:

```bash
cd HairFastGAN
pip install -r requirements.txt
pip install flask==2.2.3 flask-restx==1.1.0 Werkzeug==2.2.3
```

### 3. Download pretrained models

Download the pretrained models as described in the HairFastGAN repository:

```bash
git clone https://huggingface.co/AIRI-Institute/HairFastGAN
cd HairFastGAN && git lfs pull && cd ..
mv HairFastGAN/pretrained_models pretrained_models
mv HairFastGAN/input input
rm -rf HairFastGAN
```

## Usage

### Running the application

Start the Flask application:

```bash
python app.py
```

The application will be available at http://localhost:5000.

### API Documentation

The Swagger UI documentation is available at http://localhost:5000/api/doc

### API Endpoints

- `GET /api/hair/status` - Check the status of the model
- `POST /api/hair/generate` - Generate a new hair style
  - Required form parameters:
    - `face`: Face image file
    - `shape`: Hair shape/style image file
    - `color`: Hair color image file
  - Optional parameters:
    - `align`: Whether to align images (default: true)

## Examples

### Example usage with curl

```bash
curl -X POST http://localhost:5000/api/hair/generate \
  -F "face=@/path/to/face.png" \
  -F "shape=@/path/to/shape.png" \
  -F "color=@/path/to/color.png" \
  -F "align=true"
```

## Architecture

- `/app.py` - Main Flask application
- `/app/controllers/` - API controllers
- `/app/models/` - Model interfaces
- `/app/utils/` - Utility functions
- `/app/templates/` - HTML templates
- `/app/static/` - Static files
- `/uploads/` - Uploaded images
- `/results/` - Generated results 
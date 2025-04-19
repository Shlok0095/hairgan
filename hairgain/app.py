from flask import Flask, redirect
from flask_restx import Api
import os

# Create absolute paths for static and template folders
base_dir = os.path.dirname(os.path.abspath(__file__))
static_folder = os.path.join(base_dir, 'app', 'static')

# Create results directory in the static folder
results_dir = os.path.join(static_folder, 'results')
os.makedirs(results_dir, exist_ok=True)

# Import controllers
from app.controllers.hair_controller import api as hair_namespace

# Create Flask app
app = Flask(__name__, static_folder=static_folder)

# Configure API with swagger
authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-API-KEY'
    }
}

api = Api(
    app,
    version='1.0',
    title='HairGAN API',
    description='API for hair style generation and manipulation',
    doc='/api/doc',
    authorizations=authorizations
)

# Add namespaces
api.add_namespace(hair_namespace, path='/api/hair')

# Add a redirect from /swagger to the Swagger UI
@app.route('/swagger')
def swagger():
    return redirect('/api/doc')

if __name__ == '__main__':
    print("HairGAN API is running at http://localhost:5000")
    print("Swagger UI is available at http://localhost:5000/api/doc")
    app.run(debug=True, host='0.0.0.0', port=5000) 
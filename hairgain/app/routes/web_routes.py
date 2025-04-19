from flask import Blueprint, render_template

web_routes = Blueprint('web_routes', __name__)

@web_routes.route('/')
def index():
    return render_template('upload_form.html')

@web_routes.route('/upload')
def upload_form():
    """Render the upload form for hair generation"""
    return render_template('upload_form.html') 
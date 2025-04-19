from flask import request, url_for
from flask_restx import Namespace, Resource, fields
import werkzeug
import logging
import traceback

from app.models.hair_model import hair_model

# Create namespace
api = Namespace('hair', description='HairGAN')

# Response models
success_response = api.model('SuccessResponse', {
    'success': fields.Boolean(description='Operation was successful'),
    'result_path': fields.String(description='Path to the result file'),
    'url': fields.String(description='URL to access the result file')
})

error_response = api.model('ErrorResponse', {
    'success': fields.Boolean(description='Operation was not successful'),
    'error': fields.String(description='Error message')
})

# Parsers for file uploads
upload_parser = api.parser()
upload_parser.add_argument('face', location='files', type=werkzeug.datastructures.FileStorage, required=True, help='Image of the face')
upload_parser.add_argument('shape', location='files', type=werkzeug.datastructures.FileStorage, required=True, help='Image of the hair shape/style')
upload_parser.add_argument('color', location='files', type=werkzeug.datastructures.FileStorage, required=True, help='Image of the hair color')
upload_parser.add_argument('align', type=bool, required=False, default=True, help='Whether to align the images')
upload_parser.add_argument('blend_threshold', type=float, required=False, default=0.5, help='Threshold for blending (0.0 to 1.0)')

# Routes
@api.route('/generate')
class HairGenerate(Resource):
    @api.expect(upload_parser)
    @api.response(200, 'Success', success_response)
    @api.response(400, 'Validation Error', error_response)
    @api.response(500, 'Internal Server Error', error_response)
    def post(self):
        """
        Generate a new hair style by combining the uploaded images
        """
        try:
            args = upload_parser.parse_args()
            
            # Get the uploaded files
            face_file = args['face']
            shape_file = args['shape']
            color_file = args['color']
            align = args.get('align', True)
            blend_threshold = args.get('blend_threshold', 0.5)
            
            # Validate blend_threshold value
            if blend_threshold < 0.0 or blend_threshold > 1.0:
                return {'success': False, 'error': 'Blend threshold must be between 0.0 and 1.0'}, 400
            
            # Check if files are valid
            if not face_file or not shape_file or not color_file:
                return {'success': False, 'error': 'All image files are required'}, 400
                
            # Process the images with the hair model
            result_path, filename = hair_model.process_images(face_file, shape_file, color_file, align, blend_threshold)
            
            # Generate the URL to access the result
            result_url = url_for('static', filename=f'results/{filename}', _external=True)
            
            return {
                'success': True,
                'result_path': result_path,
                'url': result_url
            }, 200
            
        except Exception as e:
            logging.error(f"Error in hair generation: {str(e)}")
            logging.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}, 500 
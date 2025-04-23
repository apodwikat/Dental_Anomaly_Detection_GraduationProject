from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv8 model
model_path = r"C:\Users\hp\Dentist\runs\detect\train24\weights\best.pt"
model = YOLO(model_path)

# Import preprocessing utilities
from preprocessing_utils import ImageResizer, CLAHEEnhancer, preprocess_image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Read the image
            original_image = cv2.imread(filepath)
            if original_image is None:
                return jsonify({'error': 'Could not read image'}), 400

            # Preprocess the image
            preprocessed_image = preprocess_image(original_image)

            # Save preprocessed image for verification
            preprocessed_filename = f'preprocessed_{filename}'
            preprocessed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], preprocessed_filename)
            cv2.imwrite(preprocessed_filepath, cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR))

            # Run inference on preprocessed image
            results = model(preprocessed_filepath)

            # Save the annotated image
            annotated_filename = f'annotated_{filename}'
            annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
            results[0].save(annotated_filepath)

            # Process detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]

                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Get class name
                    class_name = result.names[cls]

                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })

            return jsonify({
                'success': True,
                'predictions': detections,
                'original_image': f'/static/uploads/{filename}',
                'preprocessed_image': f'/static/uploads/{preprocessed_filename}',
                'annotated_image': f'/static/uploads/{annotated_filename}'
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
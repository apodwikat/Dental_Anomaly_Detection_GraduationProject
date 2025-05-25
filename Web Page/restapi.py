from flask import Flask, request, jsonify
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

# Import preprocessing utilities
from preprocessing_utils import preprocess_image

app = Flask(__name__)

# Load YOLOv8 model
model_path = r"C:\Users\hp\Dentist\runs\detect\train24\weights\best.pt"
model = YOLO(model_path)

@app.route('/api/detect', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    
    try:
        # Read image
        image_bytes = image_file.read()
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if image_np is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Preprocess image
        preprocessed_image = preprocess_image(image_np)
        
        # Save preprocessed image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            cv2.imwrite(temp_file.name, cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR))
        
        # Run inference
        results = model(temp_file.name)
        
        # Clean up temporary file
        os.unlink(temp_file.name)

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
            'predictions': detections
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)  # Different port from main app
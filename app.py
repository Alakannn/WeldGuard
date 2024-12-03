from flask import Flask, render_template, request
import numpy as np
import base64
import cv2
import os
from ultralytics import YOLO

# Load the YOLO object detection model (using lightweight nano version)
model = YOLO('./models/welding train/runs/detect/train7/weights/best.pt')

# Set of allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize Flask application
app = Flask(__name__)

def allowed_file(filename):

    # Check if file has an extension and is in the allowed list
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_on_image(image_stream):

    # Convert image stream to numpy array for OpenCV processing
    # Uses uint8 to ensure correct image representation
    image = cv2.imdecode(
        np.asarray(bytearray(image_stream.read()), dtype=np.uint8), 
        cv2.IMREAD_COLOR
    )

    # Run object detection using YOLO model
    results = model.predict(image)

    # Annotate the image with detection results
    for i, r in enumerate(results):
        im_bgr = r.plot(conf=False)

    return im_bgr

@app.route('/', methods=['GET', 'POST'])
def home():

    # Handle form submission
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        # Get the uploaded file
        file = request.files['file']

        # Check if filename is empty
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # Validate file type and process if valid
        if file and allowed_file(file.filename):
            # Perform object detection
            predicted_image = predict_on_image(file.stream)

            # Encode detected image to base64 for HTML rendering
            retval, buffer = cv2.imencode('.png', predicted_image)
            detection_img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Reset file stream and encode original image to base64 for HTML rendering
            file.stream.seek(0)
            original_img_base64 = base64.b64encode(file.stream.read()).decode('utf-8')

            # Render results page with both original and detected images
            return render_template(
                'result.html', 
                original_img_data=original_img_base64,
                detection_img_data=detection_img_base64
            )

    # Render initial upload page for GET requests
    return render_template('index.html')

# Main entry point of the application
if __name__ == '__main__':
    # Set Flask environment to development
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=False, port=5000, host='0.0.0.0')
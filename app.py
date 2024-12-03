from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import base64
import cv2
import os
from ultralytics import YOLO
from werkzeug.security import generate_password_hash, check_password_hash

# Load the YOLO object detection model (using lightweight nano version)
model = YOLO('./models/runs/detect/train7/weights/best.pt')

# Set of allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Mock user database
users = {"admin": generate_password_hash("password")}

def allowed_file(filename):
    """Check if file has an extension and is in the allowed list"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_on_image(image_stream):
    """Run object detection on an uploaded image"""
    # Convert image stream to numpy array for OpenCV processing
    image = cv2.imdecode(
        np.asarray(bytearray(image_stream.read()), dtype=np.uint8), 
        cv2.IMREAD_COLOR
    )
    # Run object detection using YOLO model
    results = model.predict(image)
    # Annotate the image with detection results
    for r in results:
        im_bgr = r.plot(conf=False)
    return im_bgr

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['user'] = username
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout route"""
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def home():
    """Main upload page"""
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            predicted_image = predict_on_image(file.stream)
            retval, buffer = cv2.imencode('.png', predicted_image)
            detection_img_base64 = base64.b64encode(buffer).decode('utf-8')
            file.stream.seek(0)
            original_img_base64 = base64.b64encode(file.stream.read()).decode('utf-8')
            return render_template(
                'result.html',
                original_img_data=original_img_base64,
                detection_img_data=detection_img_base64
            )
    return render_template('index.html')

if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=False, port=5000, host='0.0.0.0')

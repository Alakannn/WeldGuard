from flask import Flask, render_template, request, redirect, url_for, session, Response
import numpy as np
import base64
import cv2
import os
from ultralytics import YOLO
import math

# Load the YOLO object detection model hihi
model = YOLO('./models/runs/detect/train7/weights/best.pt')

# Set of allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Dummy credentials
USER_CREDENTIALS = {
    'username': 'admin',
    'password': 'admin'
}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_on_image(image_stream):
    """Process and predict image using YOLO model."""
    image = cv2.imdecode(
        np.asarray(bytearray(image_stream.read()), dtype=np.uint8), 
        cv2.IMREAD_COLOR
    )
    results = model.predict(image)
    for i, r in enumerate(results):
        im_bgr = r.plot(conf=False)
    return im_bgr

def gen_frames():
    """Generate video frames with YOLO predictions."""
    cap = cv2.VideoCapture(0)

    classNames = ["Bad Welding", "Crack", "Excess Reinforcement",
                  "Good Welding", "Porosity", "Spatters"]

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0] * 100)) / 100  # Confidence value
                cls = int(box.cls[0])
                cv2.putText(img, classNames[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Encode the frame to JPEG and return it as a byte stream
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    """Render the home page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle the login page."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USER_CREDENTIALS['username'] and password == USER_CREDENTIALS['password']:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout the user."""
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/camera_feed')
def camera_feed():
    """Stream the camera feed with YOLO object detection."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['POST'])
def upload_file():
    """Handle file upload for YOLO predictions."""
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
    app.run(debug=True, port=5000, host='0.0.0.0')

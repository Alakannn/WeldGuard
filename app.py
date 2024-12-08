from flask import Flask, render_template, request, redirect, url_for, session, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import base64
import os
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from welding_detection import load_model, predict_on_image, generate_video_frames, allowed_file, CLASS_NAMES

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///welding_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    detections = db.relationship('Detection', backref='user', lazy=True)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    original_image = db.Column(db.Text, nullable=False)
    detected_image = db.Column(db.Text, nullable=False)
    detection_class = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

def init_model():
    try:
        return load_model()
    except Exception as e:
        app.logger.error(f"Failed to load model: {str(e)}")
        return None

model = init_model()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    return render_template('index.html', class_names=CLASS_NAMES)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            return render_template('signup.html', error='Username already exists')

        user = User(
            username=username,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()

        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = username
            return redirect(url_for('index'))

        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/camera_feed')
@login_required
def camera_feed():
    try:
        return Response(
            generate_video_frames(model),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        app.logger.error(f"Camera feed error: {str(e)}")
        return "Camera feed error", 500

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return {'error': 'No file uploaded'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': 'No selected file'}, 400

    if not allowed_file(file.filename):
        return {'error': 'Invalid file type'}, 400

    try:
        _, detection_img_base64, class_name, confidence = predict_on_image(model, file)

        file.seek(0)
        original_img_base64 = base64.b64encode(file.read()).decode('utf-8')

        detection = Detection(
            user_id=session['user_id'],
            original_image=original_img_base64,
            detected_image=detection_img_base64,
            detection_class=class_name,
            confidence=confidence
        )
        db.session.add(detection)
        db.session.commit()

        return render_template(
            'result.html',
            original_img_data=original_img_base64,
            detection_img_data=detection_img_base64,
            class_name=class_name,
            confidence=confidence,
            class_names=CLASS_NAMES
        )
    except Exception as e:
        app.logger.error(f"File processing error: {str(e)}")
        return {'error': 'Error processing file'}, 500

@app.route('/history')
@login_required
def history():
    detections = Detection.query.filter_by(user_id=session['user_id'])\
                              .order_by(Detection.timestamp.desc())\
                              .all()
    return render_template('history.html', detections=detections)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
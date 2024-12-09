from flask import Flask, render_template, request, redirect, url_for, session, Response, flash
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

# Database Models
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
        model = load_model()
        if model is None:
            raise ValueError("Model failed to load")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = init_model()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
@login_required
def index():
    return redirect(url_for('camera_feed'))

@app.route('/camera')
@login_required
def camera_feed():
    return render_template('camera_feed.html', class_names=CLASS_NAMES)

@app.route('/video_feed')
@login_required
def video_feed():
    if model is None:
        return "Model not loaded", 500
    return Response(
        generate_video_frames(model),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/upload_file', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('upload'))

    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('upload'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Please use JPG, PNG or JPEG')
        return redirect(url_for('upload'))

    try:
        # Get original image first
        file_data = file.read()
        original_img_base64 = base64.b64encode(file_data).decode('utf-8')
        
        # Reset file pointer for detection
        file.seek(0)
        
        # Perform detection
        _, detection_img_base64, class_name, confidence = predict_on_image(model, file)
        
        # Save to database
        detection = Detection(
            user_id=session['user_id'],
            original_image=original_img_base64,
            detected_image=detection_img_base64,
            detection_class=class_name,
            confidence=confidence
        )
        db.session.add(detection)
        db.session.commit()

        # Return result page
        return render_template(
            'result.html',
            original_img_data=original_img_base64,
            detection_img_data=detection_img_base64,
            class_name=class_name,
            confidence=confidence
        )
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        flash('Error processing file')
        return redirect(url_for('upload'))

@app.route('/upload', methods=['GET'])
@login_required
def upload():
    return render_template('upload.html')

@app.route('/analysis')
@login_required
def analysis():
    user_detections = Detection.query.filter_by(user_id=session['user_id']).order_by(Detection.timestamp.desc()).all()
    
    total_detections = len(user_detections)
    class_counts = {}
    confidence_data = []
    daily_counts = {}
    
    if total_detections > 0:
        # Class distribution
        for detection in user_detections:
            class_name = detection.detection_class or 'Unknown'
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        # Confidence data for trend
        confidence_data = [
            {
                'timestamp': detection.timestamp.strftime('%Y-%m-%d'),
                'confidence': float(detection.confidence) if detection.confidence else 0.0
            }
            for detection in user_detections
        ]
        
        # Daily detection counts
        for detection in user_detections:
            date = detection.timestamp.strftime('%Y-%m-%d')
            daily_counts[date] = daily_counts.get(date, 0) + 1
    
        # Calculate average confidence
        confidences = [d.confidence for d in user_detections if d.confidence]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    else:
        avg_confidence = 0
    
    return render_template(
        'analysis.html',
        total_detections=total_detections,
        class_counts=class_counts,
        confidence_data=confidence_data,
        daily_counts=daily_counts,
        avg_confidence=avg_confidence
    )

@app.route('/history')
@login_required
def history():
    detections = Detection.query.filter_by(user_id=session['user_id'])\
                              .order_by(Detection.timestamp.desc())\
                              .all()
    return render_template('history.html', detections=detections)

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
        
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        user = User(
            username=username,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out')
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database initialized")
        if model is None:
            print("WARNING: Model failed to load!")
        else:
            print("Model loaded successfully")
    app.run(debug=True, host='0.0.0.0', port=5000)

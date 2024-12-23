from flask import Flask, render_template, request, redirect, url_for, session, Response, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import base64
import os
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from welding_detection import load_model, predict_on_image, generate_video_frames, allowed_file, CLASS_NAMES
from collections import defaultdict
from sqlalchemy import func
import pytz

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///welding_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')

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
    try:
        current_time = datetime.now(malaysia_tz)
        start_time = current_time - timedelta(hours=24)
        one_hour_ago = current_time - timedelta(hours=1)
        
        user_detections = Detection.query.filter_by(user_id=session['user_id']).order_by(Detection.timestamp.desc()).all()
        
        total_detections = len(user_detections)
        class_counts = {}
        confidence_data = []
        daily_counts = {}
        latest_detection = user_detections[0] if total_detections > 0 else None

        timeline_detections = Detection.query.filter(
            Detection.user_id == session['user_id'],
            Detection.timestamp >= one_hour_ago
        ).order_by(Detection.timestamp.asc()).all()
        
        timeline_data = defaultdict(lambda: defaultdict(int))
        all_classes = set()

        if total_detections > 0:
            for detection in user_detections:
                class_name = detection.detection_class or 'Unknown'
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                malaysia_time = detection.timestamp.astimezone(malaysia_tz)
                confidence_data.append({
                    'timestamp': malaysia_time.strftime('%Y-%m-%d'),
                    'confidence': float(detection.confidence) if detection.confidence else 0.0
                })
                
            for detection in timeline_detections:
                malaysia_time = detection.timestamp.astimezone(malaysia_tz)
                timestamp = malaysia_time.strftime('%Y-%m-%d %H:%M:%S')
                class_name = detection.detection_class or 'Unknown'
                timeline_data[timestamp][class_name] += 1
                all_classes.add(class_name)
            
            for detection in user_detections:
                malaysia_time = detection.timestamp.astimezone(malaysia_tz)
                date = malaysia_time.strftime('%Y-%m-%d')
                daily_counts[date] = daily_counts.get(date, 0) + 1
            
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
            avg_confidence=avg_confidence,
            latest_detection=latest_detection,
            timeline_data=dict(timeline_data),
            all_classes=list(all_classes)
        )
    except Exception as e:
        print(f"Error in analysis route: {str(e)}")
        return "An error occurred while loading the analysis page", 500

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

@app.route('/real_time_data')
@login_required
def real_time_data():
    # Fetch real-time data for confidence values
    confidence_data = Detection.query.filter_by(user_id=session['user_id']).order_by(Detection.timestamp.desc()).limit(10).all()
    confidence_values = [{'timestamp': detection.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'confidence': detection.confidence} for detection in confidence_data]
    
    return jsonify(confidence_values)

@app.route('/welding_stats')
@login_required
def welding_stats():
    # Fetch cumulative success/failure data
    success_count = Detection.query.filter_by(detection_class='Success').count()
    failure_count = Detection.query.filter_by(detection_class='Failure').count()
    
    return jsonify({'success': success_count, 'failure': failure_count})

@app.route('/latest_detection_image', methods=['GET'])
@login_required
def latest_detection_image():
    latest_detection = Detection.query.filter_by(user_id=session['user_id']).order_by(Detection.timestamp.desc()).first()
    
    if latest_detection and latest_detection.detected_image:
        return jsonify({'detected_image': latest_detection.detected_image})
    else:
        return jsonify({'detected_image': None}), 404  # Return 404 if no image is found

def get_class_distribution():
    # Fetch the latest 100 detections
    latest_detections = Detection.query.filter_by(user_id=session['user_id']).order_by(Detection.timestamp.desc()).limit(100).all()
    
    class_counts = {}
    for detection in latest_detections:
        class_name = detection.detection_class or 'Unknown'
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return class_counts

@app.route('/latest_class_distribution', methods=['GET'])
@login_required
def latest_class_distribution():
    try:
        # Fetch the latest class distribution data
        class_counts = get_class_distribution()
        return jsonify(class_counts)
    except Exception as e:
        print(f"Error in latest_class_distribution: {str(e)}")
        return jsonify({"error": "An error occurred while fetching class distribution"}), 500

@app.route('/welding_timeline')
@login_required
def welding_timeline():
    try:
        current_time = datetime.now(malaysia_tz)
        start_time = current_time - timedelta(hours=1)
        
        current_time_utc = current_time.astimezone(pytz.UTC)
        start_time_utc = start_time.astimezone(pytz.UTC)
        
        detections = Detection.query.filter(
            Detection.user_id == session['user_id'],
            Detection.timestamp >= start_time_utc,
            Detection.timestamp <= current_time_utc
        ).order_by(Detection.timestamp.asc()).all()  # Changed to ascending order
        
        timeline_data = defaultdict(lambda: defaultdict(int))
        all_classes = set()
        
        for detection in detections:
            utc_time = pytz.UTC.localize(detection.timestamp)
            malaysia_time = utc_time.astimezone(malaysia_tz)
            
            rounded_time = malaysia_time.replace(
                second=0,
                microsecond=0,
                minute=(malaysia_time.minute // 5) * 5
            )
            timestamp = rounded_time.strftime('%Y-%m-%d %H:%M')
            class_name = detection.detection_class or 'Unknown'
            timeline_data[timestamp][class_name] += 1
            all_classes.add(class_name)
        
        timestamps = sorted(timeline_data.keys(), reverse=True)  # Reversed the order
        
        datasets = []
        colors = {
            'Good Weld': '#4e73df',
            'Bad Weld': '#e74a3b',
            'Crack': '#f6c23e',
            'Porosity': '#1cc88a',
            'Undercut': '#36b9cc',
            'Unknown': '#858796'
        }
        
        for class_name in sorted(all_classes):
            dataset = {
                'label': class_name,
                'data': [timeline_data[t][class_name] for t in timestamps],
                'borderColor': colors.get(class_name, '#000000'),
                'backgroundColor': 'rgba(0, 0, 0, 0)',
                'borderWidth': 2,
                'tension': 0.4,
                'fill': False
            }
            datasets.append(dataset)
            
        return jsonify({
            'labels': timestamps,
            'datasets': datasets
        })
        
    except Exception as e:
        print(f"Error in welding_timeline: {str(e)}")
        return jsonify({"error": "An error occurred"}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database initialized")
        if model is None:
            print("WARNING: Model failed to load!")
        else:
            print("Model loaded successfully")
    app.run(debug=True, host='0.0.0.0', port=5000)

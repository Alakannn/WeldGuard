from flask import Flask, render_template, request, redirect, url_for, session, Response
import base64
import os
from welding_detection import load_model, predict_on_image, generate_video_frames, allowed_file

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

model = load_model()

USER_CREDENTIALS = {
    'username': 'admin',
    'password': 'admin'
}

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
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
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/camera_feed')
def camera_feed():
    return Response(generate_video_frames(model), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            _, detection_img_base64 = predict_on_image(model, file.stream)
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
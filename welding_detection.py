import numpy as np
import cv2
import math
import base64
from ultralytics import YOLO

def load_model(model_path='./models/runs/detect/train7/weights/best.pt'):
    return YOLO(model_path)

CLASS_NAMES = ["Bad Welding", "Crack", "Excess Reinforcement", 
               "Good Welding", "Porosity", "Spatters"]

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_on_image(model, image_stream):
    image = cv2.imdecode(
        np.asarray(bytearray(image_stream.read()), dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    results = model.predict(image)
    for i, r in enumerate(results):
        im_bgr = r.plot(conf=False)
    return im_bgr, base64.b64encode(cv2.imencode('.png', im_bgr)[1]).decode('utf-8')

def detect_objects(model, img):
    results = model(img, stream=True, conf=0.8)
    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cv2.putText(img, CLASS_NAMES[cls], (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return img

def generate_video_frames(model):
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        annotated_img = detect_objects(model, img)

        ret, buffer = cv2.imencode('.jpg', annotated_img)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
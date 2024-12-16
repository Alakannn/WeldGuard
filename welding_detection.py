import numpy as np
import cv2
import base64
from ultralytics import YOLO

CLASS_NAMES = ["Bad Welding", "Crack", "Excess Reinforcement", 
               "Good Welding", "Porosity", "Spatters"]

def load_model(model_path='./models/runs/detect/train7/weights/best.pt'):

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def allowed_file(filename):

    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_on_image(model, image_stream):

    image_bytes = image_stream.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(image, conf=0.48)[0]

    class_name = "Unknown"
    confidence = 0.0

    if len(results.boxes) > 0:
        box = results.boxes[0]
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = CLASS_NAMES[class_id]

    # Draw results
    annotated_img = results.plot(line_width=2)

    # Convert to base64
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return annotated_img, img_base64, class_name, confidence * 100

def generate_video_frames(model):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break

        try:
            results = model.predict(frame, conf=0.5)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                           (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Frame processing error: {e}")
            continue

    cap.release()
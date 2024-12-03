from ultralytics import YOLO

model = YOLO('runs/detect/train7/weights/best.pt')
image = '/home/ubuntu/capstone/test/images/Good-Welding-images_47_jpeg_jpg.rf.82630d380c49fb0a597d7343a033ebf7.jpg'
# image = '/home/ubuntu/capstone/test/images/bad_weld_vid92_jpeg_jpg.rf.ea3c805c99ab9ff65a27c65112e22f87.jpg'

results = model.predict(source=image, save=True, show=True)
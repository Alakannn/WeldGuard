import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('best.pt')

image_path = '/home/ubuntu/Desktop/E1G2/WeldGuard/welding train/valid/images/22-possible-causes-of-weld-metal-porosity-weld-beads_jpg.rf.db243564f3559b3cb40704832e48f6eb.jpg'

results = model.predict(source=image_path, save=False)

annotated_frame = results[0].plot()

annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

plt.imshow(annotated_frame_rgb)
plt.axis('off')
plt.show()

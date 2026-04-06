import cv2
from ultralytics import YOLO

model = YOLO('models/yolo.pt')
frame = cv2.imread('test_output/sample_input.jpg')
results = model(frame)

boxes = results[0].boxes
if boxes is not None:
    print(f"YOLO detected {len(boxes)} objects.")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  Class: {model.names[cls_id]} (ID {cls_id}) | Conf: {conf:.2f} | BBox: ({int(x1)}, {int(y1)}, {int(x2-x1)}, {int(y2-y1)})")
else:
    print("YOLO detected 0 objects.")

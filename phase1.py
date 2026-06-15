from ultralytics import YOLO

# Load YOLO26 nano — auto-downloads weights on first run
model = YOLO("yolo26n.pt")

# Run inference on a test image
# Replace this with a path to any airplane image you have
results = model(r"C:\Users\p007153K\Proyecto\Data\airplane.jpg", conf = .20)

# Airplane is COCO class ID 4
AIRPLANE_CLASS_ID = 4

for result in results:
    boxes = result.boxes
    airplane_mask = boxes.cls == AIRPLANE_CLASS_ID
    airplane_boxes = boxes[airplane_mask]

    print(f"\nDetected {len(airplane_boxes)} airplane(s)")

    for box in airplane_boxes:
        conf = box.conf.item()
        xyxy = box.xyxy[0].tolist()
        print(f"  Confidence: {conf:.2f} | BBox: {[round(v) for v in xyxy]}")

# Show annotated result
results[0].show()
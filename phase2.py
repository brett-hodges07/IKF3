import cv2
import time
from ultralytics import YOLO

model = YOLO(r"yolo26s.pt")

cap = cv2.VideoCapture(0)  # Index 0 = Webcam

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Class map — press key to select which class to lock onto
CLASS_MAP = {
    ord('p'): (0,  "Person"),
    ord('a'): (4,  "Airplane"),
    ord('c'): (2,  "Car"),
    ord('f'): (67, "Phone"),
}

print("Running...")
print("  SPACE = lock onto highest confidence detection in frame")
print("  P = track Person | A = track Airplane | C = track Car | F = track Phone")
print("  Q = quit")

# Tracking state
locked_id    = None
lost_counter = 0
LOST_THRESHOLD = 30   # frames before accepting target is truly gone
PADDING        = 20   # pixels of padding around zoomed bbox, lower = more zoom

# Active class — start with person
active_class_id   = 0
active_class_name = "Person"

# FPS tracking
fps          = 0.0
frame_count  = 0
fps_start    = time.time()

# Returns cropped region of frame
def get_padded_crop(frame, x1, y1, x2, y2, padding):
    h, w = frame.shape[:2]  # Returns height and width
    x1 = max(0, int(x1) - padding)  # subtracts padding from left/top edges
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)  # adds padding to right/bottom, clamped to frame dimensions
    y2 = min(h, int(y2) + padding)
    return frame[y1:y2, x1:x2]  # Slices frame to return cropped region

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    h, w = frame.shape[:2]

    # FPS calculation — rolling average over 1 second
    frame_count += 1
    elapsed = time.time() - fps_start
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_start = time.time()

    # Run tracking, filter to active class only
    # Activates BoT-SORT — separate from YOLO detection algorithm.
    # Matches detections frame-to-frame using GMC optical flow + Kalman filter
    results = model.track(
        frame,
        persist=True,
        classes=[active_class_id],
        conf=0.35,              # lower = catches more but more noise
        verbose=False,
        iou=0.45,
        tracker="botsort.yaml",
        device='cuda'  
    )

    # Box Extraction
    boxes        = results[0].boxes  # contains xyxy coords, class ids, conf, and track ids
    annotated_frame = results[0].plot()
    display_frame   = annotated_frame.copy()

    # Lock Logic
    if boxes is not None and len(boxes) > 0 and boxes.id is not None:
        track_ids  = boxes.id.int().tolist()
        class_ids  = boxes.cls.int().tolist()
        confs      = boxes.conf.tolist()

        # Auto-lock onto first detection if no lock
        if locked_id is None:
            for i, cls in enumerate(class_ids):
                if cls == active_class_id:
                    locked_id = track_ids[i]
                    lost_counter = 0
                    print(f"Auto-locked onto ID: {locked_id} ({active_class_name})")
                    break

        # Find the locked target
        locked_found = False
        for i, tid in enumerate(track_ids):
            if tid == locked_id:
                locked_found = True
                lost_counter = 0
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = confs[i]

                # Crop and zoom into the locked target
                crop = get_padded_crop(frame, x1, y1, x2, y2, PADDING)
                if crop.size > 0:
                    display_frame = cv2.resize(crop, (w, h))

                # Draw confidence + ID overlay on zoomed frame
                cv2.putText(display_frame, f"ID: {locked_id} | Conf: {conf:.2f}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                break

        # Target not found this frame — increment counter
        if not locked_found:
            lost_counter += 1
            if lost_counter >= LOST_THRESHOLD:
                print(f"Target ID {locked_id} truly lost after {LOST_THRESHOLD} frames — resetting")
                locked_id    = None
                lost_counter = 0

    else:
        # No detections at all this frame
        lost_counter += 1
        if lost_counter >= LOST_THRESHOLD:
            locked_id    = None
            lost_counter = 0

    # --- Overlays ---

    # FPS counter (top right)
    fps_text = f"FPS: {fps:.1f}"
    fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(display_frame, fps_text, (w - fps_size[0] - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Active class (top left)
    cv2.putText(display_frame, f"Class: {active_class_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Lock status (bottom left)
    status = f"LOCKED ID {locked_id}" if locked_id else "SEARCHING..."
    color  = (0, 0, 255) if locked_id else (0, 255, 0)
    cv2.putText(display_frame, status, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("YOLO26 Zoom Lock Tracker", display_frame)

    # --- Key handling ---
    key = cv2.waitKey(1) & 0xFF

    # Q = quit
    if key == ord('q'):
        break

    # SPACE = manual lock onto highest confidence detection in frame
    if key == ord(' '):
        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            track_ids = boxes.id.int().tolist()
            class_ids = boxes.cls.int().tolist()
            confs     = boxes.conf.tolist()

            # Find highest confidence detection of active class
            best_conf = -1
            best_id   = None
            for i, cls in enumerate(class_ids):
                if cls == active_class_id and confs[i] > best_conf:
                    best_conf = confs[i]
                    best_id   = track_ids[i]

            if best_id is not None:
                locked_id    = best_id
                lost_counter = 0
                print(f"Manual lock onto ID: {locked_id} (conf: {best_conf:.2f})")
            else:
                print(f"No {active_class_name} detected to lock onto")
        else:
            print("No detections in frame")

    # Class switch keys — resets lock when switching class
    if key in CLASS_MAP:
        new_class_id, new_class_name = CLASS_MAP[key]
        if new_class_id != active_class_id:
            active_class_id   = new_class_id
            active_class_name = new_class_name
            locked_id         = None
            lost_counter      = 0
            print(f"Switched to class: {active_class_name} — lock reset")

cap.release()
cv2.destroyAllWindows()
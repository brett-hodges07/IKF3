from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\p007153K\Proyecto\yolo26s.pt")

VIDEO_PATH   = r"C:\Users\p007153K\OneDrive - Parsons FED\Pictures\Camera Roll\WIN_20260520_15_05_50_Pro.mp4"
OUTPUT_PATH  = r"C:\Users\p007153K\Proyecto\OpenImagesTraining\output1.mp4"

AIRPLANE_CLASS_ID = 67
DISPLAY_SCALE     = 0.6

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# ── VideoWriter setup ────────────────────────────────────────────────────────
orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Output resolution matches the resized inference frame
OUT_W, OUT_H = 1280, 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_src, (OUT_W, OUT_H))
# ────────────────────────────────────────────────────────────────────────────

total_frames     = 0
detected_frames  = 0
total_detections = 0

print("Running inference on video... press Q to quit")
print(f"Saving output to: {OUTPUT_PATH}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1  # fixed — was incrementing twice before

    # Resize input before inference
    frame = cv2.resize(frame, (OUT_W, OUT_H))

    results = model(
        frame,
        classes=[AIRPLANE_CLASS_ID],
        conf=0.25,
        verbose=False
    )

    boxes          = results[0].boxes
    airplane_count = 0

    if boxes is not None and len(boxes) > 0:
        airplane_mask  = boxes.cls == AIRPLANE_CLASS_ID
        airplane_boxes = boxes[airplane_mask]
        airplane_count = len(airplane_boxes)

        if airplane_count > 0:
            detected_frames  += 1
            total_detections += airplane_count

    # Annotate frame
    annotated = results[0].plot()

    # Stats overlay on full res frame before writing
    cv2.putText(annotated, f"Frame: {total_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, f"Detections this frame: {airplane_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ── Write full res frame to output file ──────────────────────────────────
    writer.write(annotated)

    # Resize only for display window
    display_h = int(annotated.shape[0] * DISPLAY_SCALE)
    display_w = int(annotated.shape[1] * DISPLAY_SCALE)
    display   = cv2.resize(annotated, (display_w, display_h))

    cv2.imshow("YOLO26 Video Test", display)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# ── Cleanup ──────────────────────────────────────────────────────────────────
cap.release()
writer.release()  # important — finalizes the video file
cv2.destroyAllWindows()

# Summary
if total_frames == 0:
    print("No frames were read — check your video path")
else:
    print(f"\nOutput saved to: {OUTPUT_PATH}")
    print(f"Frames with detections : {detected_frames} ({100 * detected_frames / total_frames:.1f}%)")
    print(f"Total airplane detections : {total_detections}")
    print(f"Avg detections per frame  : {total_detections / total_frames:.2f}")
import cv2
import time
import threading
from collections import deque
from ultralytics import YOLO
import torch
import numpy as np
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.transforms import functional as F
from filterpy.kalman import KalmanFilter

# ============================================================
# GPU config
# ============================================================
torch.cuda.set_per_process_memory_fraction(0.7)
cv2.ocl.setUseOpenCL(False)

# ============================================================
# Load models
# ============================================================
model       = YOLO(r"yolo26n.pt")
raft_weights    = Raft_Small_Weights.DEFAULT
raft_model      = raft_small(weights=raft_weights).to('cuda').eval()
raft_transforms = raft_weights.transforms()

# ============================================================
# Kalman filter — constant acceleration model
# State:       [cx, cy, vx, vy, ax, ay, w, h]
# Measurement: [cx, cy, w, h]
# cx/cy = bbox center, w/h = bbox size
# ============================================================
def make_kalman():
    kf = KalmanFilter(dim_x=8, dim_z=4)

    dt = 1.0  # 1 frame timestep

    # State transition — constant acceleration kinematics
    kf.F = np.array([
        [1, 0, dt, 0,  0.5*dt**2, 0,         0, 0],  # cx
        [0, 1, 0,  dt, 0,         0.5*dt**2,  0, 0],  # cy
        [0, 0, 1,  0,  dt,        0,           0, 0],  # vx
        [0, 0, 0,  1,  0,         dt,          0, 0],  # vy
        [0, 0, 0,  0,  1,         0,           0, 0],  # ax
        [0, 0, 0,  0,  0,         1,           0, 0],  # ay
        [0, 0, 0,  0,  0,         0,           1, 0],  # w
        [0, 0, 0,  0,  0,         0,           0, 1],  # h
    ], dtype=float)

    # Measurement matrix — we observe cx, cy, w, h directly
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=float)

    # Measurement noise — how much we trust YOLO bbox
    kf.R = np.diag([5., 5., 10., 10.])

    # Process noise — how much we trust the motion model
    kf.Q = np.diag([1., 1., 5., 5., 10., 10., 2., 2.])

    # Initial covariance — high uncertainty at start
    kf.P = np.diag([100., 100., 50., 50., 50., 50., 50., 50.])

    return kf


def bbox_to_z(x1, y1, x2, y2):
    """Convert xyxy bbox to Kalman measurement [cx, cy, w, h]."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1
    return np.array([[cx], [cy], [w], [h]], dtype=float)


def x_to_bbox(x):
    """Convert Kalman state to xyxy bbox."""
    cx, cy, w, h = x[0, 0], x[1, 0], x[6, 0], x[7, 0]
    w = max(w, 10)
    h = max(h, 10)
    return cx - w/2, cy - h/2, cx + w/2, cy + h/2


# ============================================================
# RAFT optical flow — camera motion compensation only
# ============================================================
flow_lock   = threading.Lock()
latest_flow = None
flow_thread = None
prev_frame  = None


def compute_flow(frame1, frame2):
    small1 = cv2.resize(frame1, (640, 360))
    small2 = cv2.resize(frame2, (640, 360))

    def to_tensor(f):
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        return F.to_tensor(rgb).unsqueeze(0).to('cuda')

    img1, img2 = raft_transforms(to_tensor(small1), to_tensor(small2))
    with torch.no_grad():
        flow_preds = raft_model(img1, img2)

    flow_np   = flow_preds[5].squeeze(0).permute(1, 2, 0).cpu().numpy()
    h, w      = frame1.shape[:2]
    flow_full = cv2.resize(flow_np, (w, h))
    flow_full[..., 0] *= w / 640
    flow_full[..., 1] *= h / 360
    return flow_full


def flow_worker(frame1, frame2, frame_idx):
    global latest_flow
    result = compute_flow(frame1, frame2)
    with flow_lock:
        latest_flow = (frame_idx, result)


def get_camera_motion(flow):
    """Global median flow = camera motion vector."""
    if flow is None:
        return 0.0, 0.0
    return float(np.median(flow[..., 0])), float(np.median(flow[..., 1]))


# ============================================================
# Helpers
# ============================================================
def get_padded_crop(frame, x1, y1, x2, y2, padding):
    h, w = frame.shape[:2]
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    return crop


def reset_state():
    """Return a clean slate for all tracking state."""
    return dict(
        locked_id         = None,
        lost_counter      = 0,
        is_predicting     = False,
        kf                = None,         # Kalman filter instance
        kf_initialized    = False,
        cam_vx            = 0.0,          # camera motion this frame
        cam_vy            = 0.0,
    )


# ============================================================
# Camera + class map
# ============================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

CLASS_MAP = {
    ord('p'): (0,  "Person"),
    ord('a'): (4,  "Airplane"),
    ord('c'): (2,  "Car"),
    ord('f'): (67, "Phone"),
}

print("Running...")
print("  SPACE = lock onto highest confidence detection")
print("  P = Person | A = Airplane | C = Car | F = Phone")
print("  Q = quit")

# ============================================================
# State
# ============================================================
state             = reset_state()
active_class_id   = 0
active_class_name = "Person"
LOST_THRESHOLD    = 45     # frames before full reset (~1.5s at 30fps)
PADDING           = 20     # zoom padding when locked
PRED_PADDING      = 80     # wider search window when predicting
REACQ_CONF        = 0.10   # lower conf threshold during reacquisition

fps         = 0.0
frame_count = 0
fps_start   = time.time()

# ============================================================
# Main loop
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    h, w = frame.shape[:2]
    frame_count += 1

    # --- RAFT async thread ---
    if prev_frame is not None:
        if flow_thread is None or not flow_thread.is_alive():
            flow_thread = threading.Thread(
                target=flow_worker,
                args=(prev_frame, frame.copy(), frame_count),
                daemon=True
            )
            flow_thread.start()

    with flow_lock:
        if latest_flow is not None:
            flow_age = frame_count - latest_flow[0]
            flow = latest_flow[1] if flow_age <= 3 else None
        else:
            flow = None

    prev_frame = frame.copy()

    # Camera motion compensation — shift Kalman state by global flow
    cam_vx, cam_vy = get_camera_motion(flow)
    if state['kf_initialized'] and (abs(cam_vx) > 0.5 or abs(cam_vy) > 0.5):
        state['kf'].x[0, 0] += cam_vx   # shift cx
        state['kf'].x[1, 0] += cam_vy   # shift cy

    # --- FPS ---
    elapsed = time.time() - fps_start
    if elapsed >= 1.0:
        fps         = frame_count / elapsed
        frame_count = 0
        fps_start   = time.time()

    # --- YOLO + BoT-SORT ---
    results = model.track(
        frame,
        persist  = True,
        classes  = [active_class_id],
        conf     = 0.15 if not state['is_predicting'] else REACQ_CONF,
        iou      = 0.45,
        verbose  = False,
        tracker  = "botsort.yaml",
        device   = 'cuda'
    )

    boxes           = results[0].boxes
    annotated_frame = results[0].plot()
    display_frame   = annotated_frame.copy()

    # ============================================================
    # Lock logic
    # ============================================================
    if boxes is not None and len(boxes) > 0 and boxes.id is not None:
        track_ids = boxes.id.int().tolist()
        class_ids = boxes.cls.int().tolist()
        confs     = boxes.conf.tolist()

        # Auto-lock if no current lock
        if state['locked_id'] is None:
            for i, cls in enumerate(class_ids):
                if cls == active_class_id:
                    state['locked_id']  = track_ids[i]
                    state['lost_counter'] = 0
                    state['is_predicting'] = False

                    # Init Kalman with first detection
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    state['kf'] = make_kalman()
                    state['kf'].x[:4, 0] = bbox_to_z(x1, y1, x2, y2)[:, 0]
                    state['kf'].x[6, 0]  = x2 - x1
                    state['kf'].x[7, 0]  = y2 - y1
                    state['kf_initialized'] = True
                    print(f"Auto-locked ID: {state['locked_id']} ({active_class_name})")
                    break

        # Find locked target
        locked_found = False
        for i, tid in enumerate(track_ids):
            if tid == state['locked_id']:
                locked_found           = True
                state['lost_counter']  = 0
                state['is_predicting'] = False

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = confs[i]

                # Kalman update — correct with real YOLO measurement
                state['kf'].predict()
                state['kf'].update(bbox_to_z(x1, y1, x2, y2))

                # Zoom into locked target
                crop = get_padded_crop(frame, x1, y1, x2, y2, PADDING)
                if crop is not None:
                    display_frame = cv2.resize(crop, (w, h))

                # Overlays
                cv2.putText(display_frame, f"ID: {state['locked_id']} | Conf: {conf:.2f}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Show Kalman velocity estimate
                vx = state['kf'].x[2, 0]
                vy = state['kf'].x[3, 0]
                ax = state['kf'].x[4, 0]
                ay = state['kf'].x[5, 0]
                speed = np.sqrt(vx**2 + vy**2)
                cv2.putText(display_frame,
                            f"V: ({vx:.1f}, {vy:.1f}) | A: ({ax:.1f}, {ay:.1f}) | Spd: {speed:.1f}px/f",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                break

        # --- Target not found this frame ---
        if not locked_found:
            # Grace period — if YOLO sees ANY detection of active class nearby
            # the Kalman predicted position, don't count as lost yet
            if state['kf_initialized']:
                px1, py1, px2, py2 = x_to_bbox(state['kf'].x)
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                for i, cls in enumerate(class_ids):
                    if cls == active_class_id:
                        dx1, dy1, dx2, dy2 = boxes.xyxy[i].tolist()
                        dcx = (dx1 + dx2) / 2
                        dcy = (dy1 + dy2) / 2
                        dist = np.sqrt((pcx - dcx)**2 + (pcy - dcy)**2)
                        if dist < 40 and confs[i] > 0.25:  # tighter proximity + confidence gate
                            # Close enough — treat as found, update Kalman
                            locked_found           = True
                            state['lost_counter']  = 0
                            state['is_predicting'] = False
                            state['locked_id']     = track_ids[i]  # adopt new ID
                            state['kf'].predict()
                            state['kf'].update(bbox_to_z(dx1, dy1, dx2, dy2))
                            crop = get_padded_crop(frame, dx1, dy1, dx2, dy2, PADDING)
                            if crop is not None:
                                display_frame = cv2.resize(crop, (w, h))
                            cv2.putText(display_frame,
                                        f"ID: {state['locked_id']} | Conf: {confs[i]:.2f} (reID)",
                                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                            print(f"Track ID reassigned → {state['locked_id']}")
                            break

        if not locked_found:
            state['lost_counter'] += 1

            if not state['is_predicting']:
                state['is_predicting'] = True
                print(f"Target lost — Kalman predicting | "
                      f"vx: {state['kf'].x[2,0]:.2f} vy: {state['kf'].x[3,0]:.2f} | "
                      f"ax: {state['kf'].x[4,0]:.2f} ay: {state['kf'].x[5,0]:.2f}")

            if state['kf_initialized']:
                # Kalman predict — curves naturally from acceleration state
                state['kf'].predict()

                # Decay acceleration each lost frame — prevents runaway curve prediction
                state['kf'].x[4, 0] *= 0.85  # ax
                state['kf'].x[5, 0] *= 0.85  # ay

                # Inflate uncertainty the longer we're lost
                state['kf'].P *= 1.05

                # Get predicted bbox
                px1, py1, px2, py2 = x_to_bbox(state['kf'].x)

                # Active reacquisition — run YOLO on predicted region
                search_crop = get_padded_crop(frame, px1, py1, px2, py2, PRED_PADDING)
                if search_crop is not None:
                    reacq = model(
                        search_crop,
                        conf    = REACQ_CONF,
                        classes = [active_class_id],
                        verbose = False,
                        device  = 'cuda'
                    )
                    if reacq[0].boxes is not None and len(reacq[0].boxes) > 0:
                        # Target reacquired in predicted region
                        # Remap crop coords back to full frame space
                        crop_h, crop_w = search_crop.shape[:2]
                        rx1, ry1, rx2, ry2 = reacq[0].boxes.xyxy[0].tolist()

                        # Offset back to full frame coords
                        offset_x = max(0, int(px1) - PRED_PADDING)
                        offset_y = max(0, int(py1) - PRED_PADDING)
                        rx1 += offset_x; rx2 += offset_x
                        ry1 += offset_y; ry2 += offset_y

                        # Update Kalman with reacquired measurement
                        state['kf'].update(bbox_to_z(rx1, ry1, rx2, ry2))
                        state['lost_counter'] = 0
                        state['is_predicting'] = False
                        print(f"Reacquired in predicted region!")

                # Zoom into predicted position
                crop = get_padded_crop(frame, px1, py1, px2, py2, PADDING)
                if crop is not None:
                    display_frame = cv2.resize(crop, (w, h))

                # Uncertainty indicator — grows as prediction gets stale
                uncertainty = np.sqrt(state['kf'].P[0, 0] + state['kf'].P[1, 1])
                cv2.putText(display_frame,
                            f"PREDICTING | Lost: {state['lost_counter']}/{LOST_THRESHOLD} | "
                            f"Uncertainty: {uncertainty:.1f}px",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            # Full reset if truly lost
            if state['lost_counter'] >= LOST_THRESHOLD:
                print(f"Target truly lost after {LOST_THRESHOLD} frames — resetting")
                state = reset_state()

    else:
        # No detections at all this frame
        state['lost_counter'] += 1

        if not state['is_predicting'] and state['kf_initialized']:
            state['is_predicting'] = True
            print("Total loss — Kalman predicting")

        if state['kf_initialized']:
            state['kf'].predict()
            state['kf'].x[4, 0] *= 0.85  # decay ax
            state['kf'].x[5, 0] *= 0.85  # decay ay
            state['kf'].P *= 1.05
            px1, py1, px2, py2 = x_to_bbox(state['kf'].x)

            # Active reacquisition on predicted region
            search_crop = get_padded_crop(frame, px1, py1, px2, py2, PRED_PADDING)
            if search_crop is not None:
                reacq = model(
                    search_crop,
                    conf    = REACQ_CONF,
                    classes = [active_class_id],
                    verbose = False,
                    device  = 'cuda'
                )
                if reacq[0].boxes is not None and len(reacq[0].boxes) > 0:
                    rx1, ry1, rx2, ry2 = reacq[0].boxes.xyxy[0].tolist()
                    offset_x = max(0, int(px1) - PRED_PADDING)
                    offset_y = max(0, int(py1) - PRED_PADDING)
                    rx1 += offset_x; rx2 += offset_x
                    ry1 += offset_y; ry2 += offset_y
                    state['kf'].update(bbox_to_z(rx1, ry1, rx2, ry2))
                    state['lost_counter'] = 0
                    state['is_predicting'] = False
                    print(f"Reacquired in predicted region!")

            crop = get_padded_crop(frame, px1, py1, px2, py2, PADDING)
            if crop is not None:
                display_frame = cv2.resize(crop, (w, h))

            uncertainty = np.sqrt(state['kf'].P[0, 0] + state['kf'].P[1, 1])
            cv2.putText(display_frame,
                        f"PREDICTING | Lost: {state['lost_counter']}/{LOST_THRESHOLD} | "
                        f"Uncertainty: {uncertainty:.1f}px",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        if state['lost_counter'] >= LOST_THRESHOLD:
            print("Truly lost — resetting")
            state = reset_state()

    # ============================================================
    # Overlays
    # ============================================================
    fps_text = f"FPS: {fps:.1f}"
    fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(display_frame, fps_text, (w - fps_size[0] - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(display_frame, f"Class: {active_class_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    status = f"LOCKED ID {state['locked_id']}" if state['locked_id'] else "SEARCHING..."
    color  = (0, 0, 255) if state['locked_id'] else (0, 255, 0)
    cv2.putText(display_frame, status, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("YOLO26 Kalman Tracker", display_frame)

    # ============================================================
    # Key handling
    # ============================================================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # SPACE — manual lock onto highest confidence detection
    if key == ord(' '):
        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            track_ids = boxes.id.int().tolist()
            class_ids = boxes.cls.int().tolist()
            confs     = boxes.conf.tolist()

            best_conf = -1
            best_id   = None
            best_box  = None
            for i, cls in enumerate(class_ids):
                if cls == active_class_id and confs[i] > best_conf:
                    best_conf = confs[i]
                    best_id   = track_ids[i]
                    best_box  = boxes.xyxy[i].tolist()

            if best_id is not None:
                state = reset_state()
                state['locked_id'] = best_id
                state['kf']        = make_kalman()
                x1, y1, x2, y2    = best_box
                state['kf'].x[:4, 0] = bbox_to_z(x1, y1, x2, y2)[:, 0]
                state['kf'].x[6, 0]  = x2 - x1
                state['kf'].x[7, 0]  = y2 - y1
                state['kf_initialized'] = True
                print(f"Manual lock ID: {best_id} (conf: {best_conf:.2f})")
            else:
                print(f"No {active_class_name} detected")
        else:
            print("No detections in frame")

    # Class switch
    if key in CLASS_MAP:
        new_class_id, new_class_name = CLASS_MAP[key]
        if new_class_id != active_class_id:
            active_class_id   = new_class_id
            active_class_name = new_class_name
            state             = reset_state()
            print(f"Switched to: {active_class_name}")

cap.release()
cv2.destroyAllWindows()
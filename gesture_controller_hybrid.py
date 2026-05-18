import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Optional: only if installed
try:
    import pyautogui
    HAS_PYAUTOGUI = True
except Exception:
    HAS_PYAUTOGUI = False


MODEL_PATH = Path("models/knn_model.npz")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ==============================
# Feature extraction (must match collect_dataset.py)
# ==============================
def landmarks_to_feature(landmarks):
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)

    wrist = pts[0].copy()
    pts = pts - wrist

    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale

    return pts.reshape(-1)


# ==============================
# Load kNN model (your saved npz)
# ==============================
def load_knn_model(path: Path):
    if not path.exists():
        return None

    data = np.load(path, allow_pickle=True)
    X = data["X_train"].astype(np.float32)
    y = data["y_train"].astype(np.int64)
    labels = data["labels"].tolist()  # list of label strings in id order

    return X, y, labels


def knn_predict_one(X_train, y_train, labels, x, k=7):
    if X_train is None or len(X_train) == 0:
        return "UNKNOWN"

    k = min(k, len(X_train))
    d2 = np.sum((X_train - x) ** 2, axis=1)
    nn = np.argsort(d2)[:k]
    votes = y_train[nn]

    values, counts = np.unique(votes, return_counts=True)
    pred_id = int(values[np.argmax(counts)])

    if pred_id < 0 or pred_id >= len(labels):
        return "UNKNOWN"
    return str(labels[pred_id])


# ==============================
# Rule-based classifier (simple, consistent with your classes)
# Uses distances between fingertip and MCP to estimate "extended"
# ==============================
TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
MCP = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def dist2(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return dx * dx + dy * dy + dz * dz


def finger_extended(lm, tip_id, mcp_id, th2):
    return dist2(lm[tip_id], lm[mcp_id]) > th2


def classify_rule(lm):
    # thresholds may need tiny tuning depending on camera distance
    index = finger_extended(lm, TIP["index"], MCP["index"], 0.015)
    middle = finger_extended(lm, TIP["middle"], MCP["middle"], 0.016)
    ring = finger_extended(lm, TIP["ring"], MCP["ring"], 0.016)
    pinky = finger_extended(lm, TIP["pinky"], MCP["pinky"], 0.015)

    # thumb extended check (thumb tip vs thumb base-ish landmark 2)
    thumb = dist2(lm[TIP["thumb"]], lm[2]) > 0.010

    # OPEN_PALM
    if index and middle and ring and pinky and thumb:
        return "OPEN_PALM"

    # FIST (none extended)
    if (not index) and (not middle) and (not ring) and (not pinky) and (not thumb):
        return "FIST"

    # POINT LEFT / RIGHT based on index direction from wrist
    if index and (not middle) and (not ring) and (not pinky):
        wrist = lm[0]
        idx_tip = lm[TIP["index"]]
        if idx_tip.x < wrist.x - 0.08:
            return "POINT_LEFT"
        if idx_tip.x > wrist.x + 0.08:
            return "POINT_RIGHT"
        return "POINT_RIGHT"  # default if unclear

    # THUMBS_UP (thumb extended and mostly vertical)
    if thumb and (not index) and (not middle) and (not ring) and (not pinky):
        wrist = lm[0]
        th_tip = lm[TIP["thumb"]]
        if th_tip.y < wrist.y - 0.08:
            return "THUMBS_UP"

    return "UNKNOWN"


# ==============================
# Smoothing to avoid spam
# ==============================
def stable_label(history, min_count=5):
    if not history:
        return None
    labels = list(history)
    best = max(set(labels), key=labels.count)
    return best if labels.count(best) >= min_count else None


# ==============================
# Actions (safe defaults)
# ==============================
def do_action(label):
    # Map gestures -> keyboard actions
    # You can adjust this mapping to match your demo
    if not HAS_PYAUTOGUI:
        print("[ACTION]", label, "(pyautogui not installed)")
        return

    if label in ("OPEN_PALM", "POINT_RIGHT"):
        pyautogui.press("right")  # next slide
    elif label in ("FIST", "POINT_LEFT"):
        pyautogui.press("left")   # previous slide
    # THUMBS_UP can be something optional (e.g., play/pause)
    elif label == "THUMBS_UP":
        pyautogui.press("space")


def main():
    model = load_knn_model(MODEL_PATH)
    if model is None:
        print("ERROR: knn_model.npz not found. Run train_knn.py first.")
        return

    X_train, y_train, labels = model
    print("Loaded ML model:", len(X_train), "samples | classes:", labels)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mode = "RULE"  # or "ML"
    history = deque(maxlen=10)

    last_action = None
    last_action_time = 0.0
    cooldown = 1.0  # seconds between repeated actions

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture = "NO_HAND"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if mode == "RULE":
                    gesture = classify_rule(hand_landmarks.landmark)
                else:
                    feat = landmarks_to_feature(hand_landmarks.landmark)
                    gesture = knn_predict_one(X_train, y_train, labels, feat, k=7)

                history.append(gesture)
                stable = stable_label(history, min_count=5)

                if stable and stable != "UNKNOWN":
                    now = time.time()
                    if stable != last_action or (now - last_action_time) > cooldown:
                        last_action = stable
                        last_action_time = now
                        print("[MODE]", mode, "| [STABLE]", stable)
                        do_action(stable)

            # On-screen info
            cv2.rectangle(frame, (10, 10), (630, 140), (0, 0, 0), -1)
            cv2.putText(frame, f"MODE: {mode} (press m to toggle)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Gesture: {gesture}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "ESC: quit", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hybrid Gesture Controller (RULE vs ML)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord("m"):
                mode = "ML" if mode == "RULE" else "RULE"
                history.clear()
                print("Switched mode to:", mode)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


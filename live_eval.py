import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from pathlib import Path

MODEL_PATH = Path("models/knn_model.npz")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Ground-truth keys (you press these to log the true label)
GT_KEYS = {
    ord("1"): "OPEN_PALM",
    ord("2"): "FIST",
    ord("3"): "POINT_LEFT",
    ord("4"): "POINT_RIGHT",
    ord("5"): "THUMBS_UP",
}

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
MCP = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def landmarks_to_feature(landmarks):
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    pts = pts - wrist
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale
    return pts.reshape(-1)


def load_model():
    if not MODEL_PATH.exists():
        print("Model not found. Run train_knn.py first.")
        return None, None, None
    data = np.load(MODEL_PATH, allow_pickle=True)
    X = data["X_train"].astype(np.float32)
    y = data["y_train"].astype(np.int64)
    labels = data["labels"].tolist()
    labels = [str(x) for x in labels]
    return X, y, labels


def knn_predict(X_train, y_train, labels, x, k=7):
    k = min(k, len(X_train))
    d2 = np.sum((X_train - x) ** 2, axis=1)
    nn = np.argsort(d2)[:k]
    votes = y_train[nn]
    values, counts = np.unique(votes, return_counts=True)
    pred_id = int(values[np.argmax(counts)])
    if 0 <= pred_id < len(labels):
        return labels[pred_id]
    return "UNKNOWN"


def dist2(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return dx * dx + dy * dy + dz * dz


def finger_extended(lm, tip_id, mcp_id, th2):
    return dist2(lm[tip_id], lm[mcp_id]) > th2


def classify_rule(lm):
    # Finger extension using distance thresholds
    index = finger_extended(lm, TIP["index"], MCP["index"], 0.015)
    middle = finger_extended(lm, TIP["middle"], MCP["middle"], 0.016)
    ring = finger_extended(lm, TIP["ring"], MCP["ring"], 0.016)
    pinky = finger_extended(lm, TIP["pinky"], MCP["pinky"], 0.015)

    # Thumb extended using tip to joint (landmark 2)
    thumb = dist2(lm[TIP["thumb"]], lm[2]) > 0.010

    # OPEN_PALM
    if index and middle and ring and pinky and thumb:
        return "OPEN_PALM"

    # FIST
    if (not index) and (not middle) and (not ring) and (not pinky) and (not thumb):
        return "FIST"

    # POINT LEFT / RIGHT (index extended only)
    if index and (not middle) and (not ring) and (not pinky):
        wrist = lm[0]
        tip = lm[TIP["index"]]
        if tip.x < wrist.x - 0.08:
            return "POINT_LEFT"
        if tip.x > wrist.x + 0.08:
            return "POINT_RIGHT"
        return "POINT_RIGHT"

    # THUMBS_UP (thumb extended, others closed, thumb above wrist)
    if thumb and (not index) and (not middle) and (not ring) and (not pinky):
        wrist = lm[0]
        th_tip = lm[TIP["thumb"]]
        if th_tip.y < wrist.y - 0.08:
            return "THUMBS_UP"

    return "UNKNOWN"


def stable_label(history, min_count=5):
    if not history:
        return None
    labels = list(history)
    best = max(set(labels), key=labels.count)
    if labels.count(best) >= min_count:
        return best
    return None


def main():
    X_train, y_train, labels = load_model()
    if X_train is None:
        return

    mode = "RULE"
    total = 0
    correct = 0

    history = deque(maxlen=10)
    last_stable = "NO_HAND"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    print("LIVE EVAL")
    print("  m = toggle RULE/ML")
    print("  r = reset counters")
    print("  1..5 = log ground-truth label (OPEN, FIST, LEFT, RIGHT, UP)")
    print("  ESC = quit")
    print("Make a gesture, wait until PRED stabilizes, then press 1..5 to log.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            pred = "NO_HAND"

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                if mode == "RULE":
                    pred = classify_rule(hand.landmark)
                else:
                    feat = landmarks_to_feature(hand.landmark)
                    pred = knn_predict(X_train, y_train, labels, feat, k=7)

                history.append(pred)
                st = stable_label(history, min_count=5)
                if st is not None:
                    last_stable = st
            else:
                history.clear()
                last_stable = "NO_HAND"

            acc = (correct / total * 100.0) if total > 0 else 0.0

            # UI
            cv2.rectangle(frame, (10, 10), (630, 160), (0, 0, 0), -1)
            cv2.putText(frame, "LIVE EVALUATION", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Mode: {mode} (press m)", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"PRED: {last_stable}", (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"Acc: {acc:.1f}% ({correct}/{total})   r=reset   ESC=quit",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("live_eval", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key == ord("m"):
                mode = "ML" if mode == "RULE" else "RULE"
                history.clear()
                last_stable = "NO_HAND"
                print("Switched to", mode)

            if key == ord("r"):
                total = 0
                correct = 0
                history.clear()
                last_stable = "NO_HAND"
                print("Reset counters")

            if key in GT_KEYS:
                true_label = GT_KEYS[key]
                pred_label = last_stable

                total += 1
                if pred_label == true_label:
                    correct += 1

                acc = correct / total * 100.0
                print("True:", true_label, "Pred:", pred_label, "| Acc:", f"{acc:.1f}%", f"({correct}/{total})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ==============================
# Label mapping
# ==============================

LABEL_KEYS = {
    ord("1"): "OPEN_PALM",
    ord("2"): "FIST",
    ord("3"): "POINT_LEFT",
    ord("4"): "POINT_RIGHT",
    ord("5"): "THUMBS_UP",
    ord("0"): "SKIP",
}

OUT_CSV = Path("data/gestures_dataset.csv")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ==============================
# Convert landmarks to 63 features
# ==============================

def landmarks_to_feature(landmarks):
    # 21 landmarks * 3 coordinates = 63 features

    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)

    # translate so wrist (index 0) is origin
    wrist = pts[0].copy()
    pts = pts - wrist

    # scale by distance wrist to middle_mcp (index 9)
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale

    return pts.reshape(-1)


# ==============================
# Main
# ==============================

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    rows = []
    last_capture_time = 0
    cooldown = 0.25

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

            status = "NO_HAND"
            feature = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                feature = landmarks_to_feature(hand_landmarks.landmark)
                status = "HAND_DETECTED"

            # UI panel
            cv2.rectangle(frame, (10, 10), (630, 150), (0, 0, 0), -1)

            cv2.putText(
                frame,
                "DATA COLLECTION MODE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "1:OPEN 2:FIST 3:LEFT 4:RIGHT 5:UP s:save ESC:quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Status: {status}  Samples: {len(rows)}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("collect_dataset", frame)

            key = cv2.waitKey(1) & 0xFF

            # ESC
            if key == 27:
                print("Exit without saving.")
                break

            # Save
            if key == ord("s"):
                if not rows:
                    print("No samples collected.")
                    break

                header = ["label"] + [f"f{i}" for i in range(63)]

                with OUT_CSV.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows)

                print("Saved", len(rows), "samples to", OUT_CSV.resolve())
                break

            # Record gesture
            if key in LABEL_KEYS:
                label = LABEL_KEYS[key]
                now = time.time()

                if now - last_capture_time < cooldown:
                    continue

                last_capture_time = now

                if label == "SKIP":
                    continue

                if feature is None:
                    print("No hand detected.")
                    continue

                rows.append([label] + feature.tolist())
                print("Recorded:", label, "Total:", len(rows))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


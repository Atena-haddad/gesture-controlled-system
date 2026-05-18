import time
from collections import deque
import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ----------------------------
# Landmark indices
# ----------------------------
TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
MCP = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}  # knuckles

# ----------------------------
# Distance function (squared)
# ----------------------------
def dist2(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return dx*dx + dy*dy + dz*dz

def finger_extended_by_distance(lm, tip_id, mcp_id, threshold2):
    return dist2(lm[tip_id], lm[mcp_id]) > threshold2

# ----------------------------
# Gesture classification (v2)
# ----------------------------
def classify_gesture(lm):

    TH_INDEX = 0.015
    TH_MIDDLE = 0.016
    TH_RING = 0.016
    TH_PINKY = 0.015

    index = finger_extended_by_distance(lm, TIP["index"], MCP["index"], TH_INDEX)
    middle = finger_extended_by_distance(lm, TIP["middle"], MCP["middle"], TH_MIDDLE)
    ring = finger_extended_by_distance(lm, TIP["ring"], MCP["ring"], TH_RING)
    pinky = finger_extended_by_distance(lm, TIP["pinky"], MCP["pinky"], TH_PINKY)

    thumb_extended = dist2(lm[TIP["thumb"]], lm[2]) > 0.010

    if index and middle and ring and pinky and thumb_extended:
        return "OPEN_PALM"

    if (not index) and (not middle) and (not ring) and (not pinky):
        return "FIST"

    if index and (not middle) and (not ring) and (not pinky):
        wrist = lm[0]
        idx_tip = lm[TIP["index"]]
        if idx_tip.x > wrist.x + 0.08:
            return "POINT_RIGHT"
        if idx_tip.x < wrist.x - 0.08:
            return "POINT_LEFT"
        return "POINT"

    if thumb_extended and (not index) and (not middle) and (not ring) and (not pinky):
        thumb_tip = lm[TIP["thumb"]]
        wrist = lm[0]
        if thumb_tip.y < wrist.y - 0.08:
            return "THUMBS_UP"
        if thumb_tip.y > wrist.y + 0.08:
            return "THUMBS_DOWN"

    return "UNKNOWN"

# ----------------------------
# Temporal smoothing
# ----------------------------
def stable_label(history, min_count=5):
    if not history:
        return None
    labels = list(history)
    best = max(set(labels), key=labels.count)
    return best if labels.count(best) >= min_count else None

# ----------------------------
# Main
# ----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

history = deque(maxlen=10)
last_action = None
last_action_time = 0.0

frame_count = 0
start_time = time.time()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Could not read from camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "NO_HAND"
        action_text = ""

        if result.multi_hand_landmarks:
            hand_lms = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            gesture = classify_gesture(hand_lms.landmark)
            history.append(gesture)

            stable = stable_label(history, min_count=5)

            if stable and stable != "UNKNOWN":
                action_text = stable
                now = time.time()

                if stable != last_action or (now - last_action_time) > 1.2:
                    last_action = stable
                    last_action_time = now

                    print(f"[ACTION] {stable}")

                    # --- Presentation Control ---
                    if stable == "OPEN_PALM":
                        pyautogui.press("right")
                    elif stable == "FIST":
                        pyautogui.press("left")

        # FPS + latency
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        frame_time_ms = (time.time() - frame_start) * 1000

        cv2.rectangle(frame, (10, 10), (620, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}  Latency: {frame_time_ms:.1f} ms", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Gesture: {gesture}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("Gesture Controller - Presentation Mode", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

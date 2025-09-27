import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap_webcam = cv2.VideoCapture(0)
cap_video = cv2.VideoCapture("video.mp4")

paused = False
action_text = "Playing"
frame_skip = 30
last_action_time = 0
action_delay = 1.0  # seconds debounce

def detect_gesture(lm, handedness):
    """Very clear, non-overlapping gesture rules."""
    # Y positions (lower value = higher on screen)
    thumb_tip  = lm[4].y
    index_tip  = lm[8].y
    middle_tip = lm[12].y
    ring_tip   = lm[16].y
    pinky_tip  = lm[20].y

    thumb_ip   = lm[3].y
    index_mcp  = lm[5].y
    middle_mcp = lm[9].y
    ring_mcp   = lm[13].y
    pinky_mcp  = lm[17].y

    label = handedness.classification[0].label  # "Left" or "Right"

    # Palm open → Play
    if (thumb_tip < thumb_ip and index_tip < index_mcp and
        middle_tip < middle_mcp and ring_tip < ring_mcp and pinky_tip < pinky_mcp):
        return "Play"

    # Fist → Pause
    if (thumb_tip > thumb_ip and index_tip > index_mcp and
        middle_tip > middle_mcp and ring_tip > ring_mcp and pinky_tip > pinky_mcp):
        return "Pause"

    # Thumb up → Volume Up
    if thumb_tip < thumb_ip and index_tip > index_mcp and middle_tip > middle_mcp:
        return "IncreaseVolume"

    # Thumb down → Volume Down
    if thumb_tip > thumb_ip and index_tip > index_mcp and middle_tip > middle_mcp:
        return "DecreaseVolume"

    # Index only up → Forward/Rewind
    if index_tip < index_mcp and middle_tip > middle_mcp and ring_tip > ring_mcp and pinky_tip > pinky_mcp:
        if label == "Left":
            return "Forward10"
        else:
            return "Rewind10"

    return None


while True:
    ret_webcam, frame_webcam = cap_webcam.read()
    if not ret_webcam:
        break
    frame_webcam = cv2.flip(frame_webcam, 1)
    rgb = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = None
    if results.multi_hand_landmarks:
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame_webcam, lm, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(lm.landmark, handedness)
            cv2.putText(frame_webcam, f"Hand: {handedness.classification[0].label}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    # Only trigger if enough time has passed
    current_time = time.time()
    if gesture and current_time - last_action_time > action_delay:
        last_action_time = current_time

        if gesture == "Pause":
            paused = True
            action_text = "Paused"
        elif gesture == "Play":
            paused = False
            action_text = "Playing"
        elif gesture == "Forward10":
            f = cap_video.get(cv2.CAP_PROP_POS_FRAMES)
            cap_video.set(cv2.CAP_PROP_POS_FRAMES, f + frame_skip)
            action_text = "Forward 10s"
        elif gesture == "Rewind10":
            f = cap_video.get(cv2.CAP_PROP_POS_FRAMES)
            cap_video.set(cv2.CAP_PROP_POS_FRAMES, max(f - frame_skip, 0))
            action_text = "Rewind 10s"
        elif gesture == "IncreaseVolume":
            action_text = "Volume Up"
        elif gesture == "DecreaseVolume":
            action_text = "Volume Down"

    # Show video frame
    if not paused:
        ret_video, frame_video = cap_video.read()
        if not ret_video:
            cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
    else:
        f = cap_video.get(cv2.CAP_PROP_POS_FRAMES)
        cap_video.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ret_video, frame_video = cap_video.read()

    cv2.putText(frame_video, f"Action: {action_text}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow("Webcam", frame_webcam)
    cv2.imshow("Video", frame_video)

    if cv2.waitKey(10) & 0xFF == 27:  # ESC
        break

cap_webcam.release()
cap_video.release()
cv2.destroyAllWindows()

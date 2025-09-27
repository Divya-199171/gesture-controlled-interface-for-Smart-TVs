#pip install mediapipe opencv-python numpy
 
import cv2
import mediapipe as mp
import time
 
# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
 
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
 
# Load video
video_path = "video.mp4"
cap_video = cv2.VideoCapture(video_path)
 
# Webcam for gesture control
cap_webcam = cv2.VideoCapture(0)
 
# Playback control variables
paused = False
action_text = "Playing"
frame_skip = 30  # frames to skip for forward/rewind
 
def detect_gesture(landmarks):
    """Detect simple gestures based on finger positions."""
    thumb_tip = landmarks[4].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    ring_tip = landmarks[16].y
    pinky_tip = landmarks[20].y
 
    thumb_mcp = landmarks[2].y
    index_mcp = landmarks[5].y

    # Example gestures
    if index_tip < index_mcp and middle_tip < index_mcp:
        return "Pause"
    elif index_tip < index_mcp and middle_tip > index_mcp:
        return "Play"
    elif thumb_tip < thumb_mcp:
        return "Forward"
    elif thumb_tip > thumb_mcp:
        return "Rewind"
    return None

while True:
    # Read webcam frame
    ret_webcam, frame_webcam = cap_webcam.read()
    if not ret_webcam:
        break
    frame_webcam = cv2.flip(frame_webcam, 1)
    rgb_webcam = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_webcam)
 
    gesture = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame_webcam, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks.landmark)
 
    # Apply gesture actions
    if gesture == "Pause":
        paused = True
        action_text = "Paused"
    elif gesture == "Play":
        paused = False
        action_text = "Playing"
    elif gesture == "Forward":
        current_frame = cap_video.get(cv2.CAP_PROP_POS_FRAMES)
        cap_video.set(cv2.CAP_PROP_POS_FRAMES, current_frame + frame_skip)
        action_text = "Forward"
    elif gesture == "Rewind":
        current_frame = cap_video.get(cv2.CAP_PROP_POS_FRAMES)
        cap_video.set(cv2.CAP_PROP_POS_FRAMES, max(current_frame - frame_skip, 0))
        action_text = "Rewind"
 
    # Read video frame
    if not paused:
        ret_video, frame_video = cap_video.read()
        if not ret_video:
            cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue
    else:
        # If paused, don't advance frame
        ret_video, frame_video = cap_video.read()
        cap_video.set(cv2.CAP_PROP_POS_FRAMES, cap_video.get(cv2.CAP_PROP_POS_FRAMES) - 1)
 
    # Show action text on video
    cv2.putText(frame_video, f"Action: {action_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
 
    # Display both video and webcam
    cv2.imshow("Gesture Control - Webcam", frame_webcam)
    cv2.imshow("Video Player", frame_video)
 
    if cv2.waitKey(10) & 0xFF == 27:  # ESC to quit
        break
 
cap_webcam.release()
cap_video.release()
cv2.destroyAllWindows()
 
 
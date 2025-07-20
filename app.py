import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import subprocess

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up hand tracking with 1 hand max and a good detection threshold
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Variables to track the last performed gesture and its timestamp
prev_action = ""
last_action_time = time.time()

# Delay between repeated actions to avoid spamming
ACTION_DELAY = 1.5

# Get screen resolution (used for scrolling)
screen_height, screen_width = pyautogui.size()

# ---------------------- Finger Detection Logic ----------------------

def fingers_up(hand_landmarks):
    """
    Returns a list of booleans representing which fingers are up.
    Thumb logic is based on x-coordinates due to its side position.
    Other fingers use y-coordinates.
    """
    tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    fingers = []
    landmarks = hand_landmarks.landmark

    # Thumb: compare x coordinates
    fingers.append(landmarks[tips_ids[0]].x < landmarks[tips_ids[0] - 1].x)

    # Other four fingers: compare y coordinates
    for i in range(1, 5):
        fingers.append(landmarks[tips_ids[i]].y < landmarks[tips_ids[i] - 2].y)

    return fingers

# ---------------------- Gesture Recognition ----------------------

def detect_gesture(fingers):
    """
    Maps the current finger state to a gesture name.
    """
    if fingers == [False, True, True, False, False]:   # Peace ✌️
        return "play_pause"
    elif fingers == [False, True, False, False, False]:  # Only index finger
        return "switch_app"
    elif fingers == [True, True, True, True, True]:     # All fingers up
        return "scroll_up"
    elif fingers == [False, False, False, False, False]:  # Fist
        return "scroll_down"
    else:
        return ""

# ---------------------- Action Performer ----------------------

def perform_action(gesture):
    """
    Executes an OS-level action based on the detected gesture.
    Adds a delay between repeated actions.
    """
    global prev_action, last_action_time
    current_time = time.time()

    # Debounce mechanism to avoid repeated actions
    if gesture == prev_action and (current_time - last_action_time) < ACTION_DELAY:
        return

    # Map gesture to action
    if gesture == "play_pause":
        pyautogui.press("k")  # Simulates 'k' key for YouTube play/pause
        prev_action = "play_pause"

    elif gesture == "switch_app":
        # CMD + Tab (macOS) for switching apps
        subprocess.run(["osascript", "-e", 'tell application "System Events" to key code 48 using {command down}'])
        prev_action = "switch_app"

    elif gesture == "scroll_up":
        pyautogui.scroll(30)
        prev_action = "scroll_up"

    elif gesture == "scroll_down":
        pyautogui.scroll(-30)
        prev_action = "scroll_down"

    else:
        prev_action = ""

    last_action_time = current_time

# ---------------------- Main Loop ----------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for natural webcam mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB as MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    gesture_text = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on screen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect which fingers are up
            finger_states = fingers_up(hand_landmarks)

            # Map fingers to gesture
            gesture = detect_gesture(finger_states)

            # Perform action based on gesture
            perform_action(gesture)

            # Set display text for the gesture
            if gesture == "play_pause":
                gesture_text = "🎵 Play/Pause"
            elif gesture == "switch_app":
                gesture_text = "🔄 Switch App"
            elif gesture == "scroll_up":
                gesture_text = "⬆️ Scroll Up"
            elif gesture == "scroll_down":
                gesture_text = "⬇️ Scroll Down"
            else:
                gesture_text = ""

    # Display current gesture action on screen
    if gesture_text:
        cv2.putText(frame, gesture_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Show the camera feed
    cv2.imshow("Gesture Control", frame)

    # Quit the app by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

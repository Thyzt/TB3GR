import cv2
 
import keyboard
 
import mediapipe as mp
# Necessary imports
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
# Hand initialization. Currently, it is set to 1 hand
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Ensure you have the haarcascade_frontalface_default.xml file in the same directory or provide the correct path
 
# Defines which fingers are up and down
def count_fingers(hand_landmarks, handedness_label):
    # returns whether or not a finger is open (1) or closed (0)

    fingers = []
 
    tips = [4, 8, 12, 16, 20]
     # Thumb, index, middle, etc.
    if handedness_label == "Right":
        fingers.append(int(hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x))
  # Each knuckle + tip of the finger is assigned a number increasing from the thumb to the pinky. 
 # This says that if the tip of the finger is recognized as below the knuckle, then the finger would be counted as "down
    else:
        fingers.append(int(hand_landmarks.landmark[tips[0]].x > hand_landmarks.landmark[tips[0] - 1].x))
 
    for i in range(1, 5):
        fingers.append(int(hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y))
 # Same with thumb
    return fingers
 
# Defines the gesture returned by hand states (ex. fist means all fingers are down, five means all fingers are up)
def gesture(finger_states):
    if finger_states == [0, 0, 0, 0, 0]:
        return "fist"
    if finger_states == [1, 1, 1, 1, 1]:
        return "five"
    if finger_states == [0, 1, 0, 0, 0]:
        return "one"
    if finger_states == [0, 1, 1, 0, 0]:
        return "two"
    if finger_states == [1, 0, 0, 0, 1]:
        return "call me"
    else:
        return "No Gesture" 
 
which_hand = "Non-Existent"

# Runs the camera to capture video
cap = cv2.VideoCapture(0)
 
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
 
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
 
    frame = cv2.flip(frame, 1)  # Flips the frame horizontally for a mirror effect
    # Process the frame for hand detection
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 
    what_gesture = "idk"  # Default gesture
    # If hands are detected, draw landmarks and count fingers
 
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # get hand label
            handedness_label = results.multi_handedness[i].classification[0].label
            which_hand = handedness_label
            # count fingers and classify gestures
            finger_states = count_fingers(hand_landmarks, handedness_label)
            what_gesture = gesture(finger_states)
 
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # detect fingers
            h, w, _ = frame.shape
 
            locx = int(hand_landmarks.landmark[0].x * w)
            locy = int(hand_landmarks.landmark[0].y * h)
         # This was never used ^^. Theoretically we could have captioned each hand with the gesture at the hand's location
 
 
    # Returns a keystroke based on the current gesture detected (w = forward, x = back, d = right rotation, a = left rotation, s = stop)
    # TurtleBot3 Teleoperation mode must be active and focused on the screen/vm/pc in order for it to actually work
    if what_gesture == "five": keyboard.press_and_release('w')
    if what_gesture == "fist": keyboard.press_and_release('x')
    if what_gesture == "one": keyboard.press_and_release('d')
    if what_gesture == "two": keyboard.press_and_release('a')
    if what_gesture == "call me": keyboard.press_and_release('s')
 
    # Detects faces in the frame. This code also captures video from the default camera and displays it in a window.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Draw rectangles around detected faces
    cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # Draws telemetry text
    cv2.putText(frame, f'{which_hand} Gesture: {what_gesture}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 
    cv2.imshow('Video Feed', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
# Press 'q' to exit the video feed.

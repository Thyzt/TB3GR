import cv2
print(cv2.__version__)

import mediapipe as mp

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Ensure you have the haarcascade_frontalface_default.xml file in the same directory or provide the correct path.

# Defines which fingers are up and down
def count_fingers(hand_landmarks):
    # returns whether or not a finger is open (1) or closed (0)
    # Thumb, index, middle, etc.
    fingers = []

    tips = [4, 8, 12, 16, 20]

    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)

    else:
        fingers.append(0)

    for i in range (1, 5):
        if hand_landmarks.landmark[tips[i]].x < hand_landmarks.landmark[tips[i] - 2].x:
            fingers.append(1)

        else:
            fingers.append(0)
    
    return fingers

def gesture(finger_states):
    if finger_states == [1, 1, 1, 1, 1]:
        return "fist"
    if finger_states == [0, 0, 0, 0, 0]:
        return "five"
    #if finger_states == [0, 1, 1, 1, 1]:
    #    return "thumbs up"
    if finger_states == [1, 0, 1, 1, 1]:
        return "one"
    if finger_states == [0, 1, 1, 1, 0]:
        return "call me"
    else:
        return "idk"



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)



if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
    # Process the frame for hand detection
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    what_gesture = "idk"  # Default gesture
    # If hands are detected, draw landmarks and count fingers

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # detect fingers

            finger_states = count_fingers(hand_landmarks)
            what_gesture = gesture(finger_states)


    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Draw rectangles around detected faces
    cv2.putText(frame, f'Idiots: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Idiot helper: {what_gesture}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Video Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
# This code captures video from the default camera and displays it in a window.
# Press 'q' to exit the video feed.

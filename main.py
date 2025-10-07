# main.py
import cv2
from gestures import HandDetector
from actions import perform_action

detector = HandDetector()

cap = cv2.VideoCapture(0)
print("Camera started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    results = detector.detect(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            detector.draw(frame, results)
            finger_states = detector.get_finger_states(hand_landmarks, w, h)
            gesture = detector.classify(finger_states)
            print("Detected:", gesture)
            perform_action(gesture)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

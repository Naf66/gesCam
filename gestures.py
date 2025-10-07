# gestures.py
import mediapipe as mp
import cv2

class HandDetector:
    def __init__(self, detection_conf=0.7, tracking_conf=0.7, max_num_hands=1):
        """
        Wraps MediaPipe Hands.
        detection_conf/tracking_conf: floats between 0-1 for model confidence.
        max_num_hands: how many hands to detect (1 is simpler).
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        """
        Run MediaPipe on a BGR OpenCV frame.
        Returns the raw `results` object from MediaPipe (contains landmarks).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw(self, frame, results):
        """Draw landmarks (if any) onto the frame in-place."""
        if not results or not results.multi_hand_landmarks:
            return frame
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def landmarks_to_pixel_list(self, hand_landmarks, frame_w, frame_h):
        """
        Convert normalized landmarks to a list of (x,y) pixel tuples.
        Index of landmark corresponds to MediaPipe's 0..20 mapping.
        """
        lm_list = []
        for lm in hand_landmarks.landmark:
            lm_list.append((int(lm.x * frame_w), int(lm.y * frame_h)))
        return lm_list

    def get_finger_states(self, hand_landmarks, frame_w, frame_h):
        """
        Returns list [thumb, index, middle, ring, pinky] where 1 = finger is up/extended, 0 = down/curled.
        Simple heuristic:
          - For index/middle/ring/pinky compare tip y vs pip y (tip up if tip.y < pip.y).
          - For thumb compare tip x vs ip x (works for one orientation; see notes).
        """
        lm = self.landmarks_to_pixel_list(hand_landmarks, frame_w, frame_h)
        if len(lm) != 21:
            return [0, 0, 0, 0, 0]

        # Thumb: tip=4, ip=3 (compare x; may need flipping for left hand)
        thumb_is_open = 1 if lm[4][0] < lm[3][0] else 0

        # For other fingers, tips: 8,12,16,20 and compare with tip-2 (6,10,14,18)
        fingers = []
        tips = [8, 12, 16, 20]
        for t in tips:
            # if tip y is above (smaller) than the joint two levels down -> finger is up
            fingers.append(1 if lm[t][1] < lm[t - 2][1] else 0)

        # order: [thumb, index, middle, ring, pinky]
        return [thumb_is_open] + fingers

    def classify(self, finger_states):
        """
        Simple rule-based classifier:
         - [0,0,0,0,0] -> Fist
         - [1,1,1,1,1] -> Palm
         - [0,1,0,0,0] -> One finger (index)
         - else -> Unknown
        """
        if finger_states == [0, 0, 0, 0, 0]:
            return "Fist"
        if finger_states == [1, 1, 1, 1, 1]:
            return "Palm"
        if finger_states == [0, 1, 0, 0, 0]:
            return "OneFinger"
        return "Unknown"

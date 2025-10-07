# actions.py
import pyautogui
import time

last_action_time = 0
cooldown = 1.5  # seconds

def perform_action(gesture):
    global last_action_time
    if time.time() - last_action_time < cooldown:
        return
    last_action_time = time.time()

    if gesture == "Palm":
        print("▶️ Play/Pause")
        pyautogui.press("space")

    elif gesture == "Fist":
        print("🔉 Volume Down")
        pyautogui.press("volumedown")

    elif gesture == "OneFinger":
        print("⏭️ Next")
        pyautogui.press("right")

    elif gesture == "ThumbsUp":
        print("🔊 Volume Up")
        pyautogui.press("volumeup")

    else:
        print("Unknown Gesture")

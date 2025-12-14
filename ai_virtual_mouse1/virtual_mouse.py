import cv2
import mediapipe as mp
import pyautogui
import time

# Screen size
screen_width, screen_height = pyautogui.size()

# Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# WebCam
cap = cv2.VideoCapture(0)

# Smooth cursor movement
prev_x, prev_y = 0, 0
smoothening = 6

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                landmark_list.append([id, int(lm.x * w), int(lm.y * h)])

            # Index finger tip
            x1, y1 = landmark_list[8][1], landmark_list[8][2]

            # Middle finger tip
            x2, y2 = landmark_list[12][1], landmark_list[12][2]

            # Move cursor
            target_x = screen_width * (x1 / w)
            target_y = screen_height * (y1 / h)

            # Smooth cursor
            curr_x = prev_x + (target_x - prev_x) / smoothening
            curr_y = prev_y + (target_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Clicking gestures
            # 1) LEFT CLICK → Index & Thumb pinching
            thumb_x, thumb_y = landmark_list[4][1], landmark_list[4][2]
            distance_left = abs(x1 - thumb_x)

            if distance_left < 25:
                pyautogui.click()
                cv2.putText(img, 'Left Click', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                time.sleep(0.2)

            # 2) RIGHT CLICK → Index & Middle finger close
            distance_right = abs(x1 - x2)

            if distance_right < 20:
                pyautogui.rightClick()
                cv2.putText(img, 'Right Click', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                time.sleep(0.2)

    cv2.imshow("AI Virtual Mouse", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import serial
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

arduino = serial.Serial(port='COM5', baudrate=115200, timeout=.1)

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        finger_tips = [8, 12, 16, 20]
        finger_count = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in finger_tips:
                    if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i-2].y:
                        finger_count.append(1)
                    else:
                        finger_count.append(0)
                #print(finger_count)
                count = finger_count.count(1)
                print(count)
                arduino.write(bytes(str(count), 'utf-8'))
                time.sleep(0.020)
                """for id, lm in enumerate(hand_landmarks.landmark):
                    pixelCords = mp_drawing._normalized_to_pixel_coordinates(lm.x, lm.y, image.shape[1], image.shape[0])
                    print(id, pixelCords)
                    if pixelCords != None:
                        if id == 12:
                            cv2.circle(image, (pixelCords[0], pixelCords[1]), 10, (255, 0, 0), cv2.FILLED)"""
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Image', image)
        cv2.waitKey(1)

cap.release()

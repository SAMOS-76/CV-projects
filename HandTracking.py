import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    pixelCords = mp_drawing._normalized_to_pixel_coordinates(lm.x, lm.y, image.shape[1], image.shape[0])
                    print(id, pixelCords)
                    if pixelCords != None:
                        if id == 12:
                            cv2.circle(image, (pixelCords[0], pixelCords[1]), 10, (255, 0, 0), cv2.FILLED)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Image', image)
        cv2.waitKey(1)
    
cap.release()

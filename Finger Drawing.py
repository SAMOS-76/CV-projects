import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
redCoords = []
colour = (0, 0, 0)
pt1_x, pt1_y = 0, 0
drawing = False
erase = False
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

def drawWindow(image):
    red = cv2.rectangle(image, (0, 0), (200, 100), (0, 0, 255), cv2.FILLED)
    green = cv2.rectangle(image, (200, 0), (400, 100), (0, 255, 0), cv2.FILLED)
    blue = cv2.rectangle(image, (400, 0), (600, 100), (255, 0, 0), cv2.FILLED)
    eraser = cv2.rectangle(image, (600, 0), (800, 100), (0, 0, 0,), cv2.FILLED)
    cv2.imshow('Image', red)
    cv2.imshow('Image', green)
    cv2.imshow('Image', blue)


with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        finger_tips = [8, 12]
        finger_count = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x, y = hand_landmarks.landmark[finger_tips[0]].x, hand_landmarks.landmark[finger_tips[0]].y
                pixelCords = mp_drawing._normalized_to_pixel_coordinates(x, y, image.shape[1], image.shape[0])
                if pixelCords != None:
                    x, y = pixelCords[0], pixelCords[1]
                    for i in finger_tips:
                        if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i-2].y:
                            finger_count.append(1)
                        else:
                            finger_count.append(0)
                    print(x)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        count = finger_count.count(1)
        if count == 2:
            pt1_x, pt1_y = 0, 0
            #drawing = False
            if y < 100:
                if x < 200:
                    erase = False
                    colour = (0, 0, 255)
                elif x < 400 and x > 200:
                    erase = False
                    colour = (0, 255, 0)
                elif x < 600 and x > 400:
                    erase = False
                    colour = (255, 0, 0)
                elif x < 800 and x >600:
                    colour = (0, 0, 0,)
                    erase = True
        elif count == 1:
            cv2.circle(image, (x, y), 15, colour, cv2.FILLED)
            if pt1_x==0 and pt1_y == 0:
                pt1_x, pt1_y =  x, y
            #cv2.line(image, (pt1_x, pt1_y), (x, y), colour, thickness=3)
            if erase == True:
                cv2.line(imgCanvas, (pt1_x, pt1_y), (x, y), colour, thickness=35)
            else:
                cv2.line(imgCanvas, (pt1_x, pt1_y), (x, y), colour, thickness=3)
            pt1_x, pt1_y = x, y

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
        image = cv2.bitwise_and(image, imgInverse)
        image = cv2.bitwise_or(image, imgCanvas)

        #image = cv2.addWeighted(image, 0.5, imgCanvas, 0.5, 0)
        cv2.imshow('Image', image)
        cv2.imshow('Canvas', imgCanvas)
        drawWindow(image)
        cv2.waitKey(1)

cap.release()

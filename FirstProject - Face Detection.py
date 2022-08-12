import cv2

cap = cv2.VideoCapture(0)  # Laptop camera
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)
faceCascade = cv2.CascadeClassifier('Tutorial/Resources/haarcascade_frontalface_default.xml')

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', imgGray)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

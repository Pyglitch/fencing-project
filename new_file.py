import cv2
cap = cv2.VideoCapture("C:/Users/flora/Documents/FencingProject/flech_Sam.mp4")
print("Is opened:", cap.isOpened())
ret, frame = cap.read()
print("Frame read:", ret)

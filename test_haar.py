import cv2
print(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
print(cascade.empty())

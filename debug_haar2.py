import cv2

frame = cv2.imread('test_output/sample_input.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

x, y, w, h = 15, 12, 578, 365
roi = gray[y:y+h, x:x+w]

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_license_plate_rus_16stages.xml')

for neighbors in range(1, 6):
    for scale in [1.05, 1.1, 1.15]:
        plates = cascade.detectMultiScale(roi, scaleFactor=scale, minNeighbors=neighbors, minSize=(15, 15))
        if len(plates) > 0:
            print(f"16stages: Found {len(plates)} plate(s) with scaleFactor={scale}, minNeighbors={neighbors}")

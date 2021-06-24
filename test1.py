import cv2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture('steve.mp4')


while True:
   
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eye = eye_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        for (x, y, w, h) in eye:
            cv2.rectangle(img, (x, y), (x+w, y+h), (100, 0, 0), 2)
            cv2.putText(img, 'eye', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,10), 2)
        
    
    cv2.imshow('img', img)
    
    if(cv2.waitKey(1)&0xFF==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
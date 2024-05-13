import tensorflow.keras as keras
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
camera = cv2.VideoCapture(0)

model = keras.models.load_model('facemodel.h5')
model.compile(loss=keras.losses.BinaryCrossentropy(),
              optimizer='sgd',
              metrics=[keras.metrics.BinaryCrossentropy()])
cv2.namedWindow("frame",flags=cv2.WINDOW_NORMAL)
while True:
    _, frame = camera.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=3, minSize = (100,100))
    for(x, y, w, h) in face:
        #color = (255,255,255)
        stroke = 1
        reg_color = (0,0,255)
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x,y), (width, height), reg_color, stroke)
        #print((x,y,w,h))
        img = frame[y:height,x:width]
        #cv2.imshow('img', img)
        img = cv2.resize(img, (75, 75))
        img = np.reshape(img, [1, 75, 75, 3])
        #print(model.predict(img)[0][0])
        if model.predict(img)[0][0]>0.5:
            frame = cv2.putText(frame, 'Tidak Menggunakan Masker', (x, y), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            frame = cv2.putText(frame, 'Menggunakan Masker', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()

#img = cv2.imread("wear1.jpg")

#cv2.waitKey(0)

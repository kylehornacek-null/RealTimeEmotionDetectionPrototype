import cv2
import keras
from keras.preprocessing import image
import numpy as np

model = keras.models.load_model('model.h5')

haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detected = haar_cascade.detectMultiScale(img_gray, 1.32, 5)

    pixel_error = 5
    for (x, y, w, h) in face_detected:
        cv2.rectangle(frame, (x + pixel_error, y + pixel_error), (x + w + pixel_error, y + h + pixel_error), (53, 53, 198), thickness=3)
        roi_gray = img_gray[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        max = np.argmax(predictions[0])

        emotion = ['Happy', 'Sad', 'Angry']
        predicted_emotion = emotion[max]

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, 1, (53, 53, 198), 2)

        # show the frame on the newly created image window
    cv2.imshow('Frames', frame)

    # this condition is used to run a frames at the interval of 10 mili sec
    # and if in b/w the frame running , any one want to stop the execution .
    if cv2.waitKey(10) & 0xFF == ord('q'):
        # break out of the while loop
        break
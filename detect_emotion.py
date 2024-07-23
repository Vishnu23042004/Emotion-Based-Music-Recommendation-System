# import required packages
import sys
import io
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

emotion_model = Sequential([
    InputLayer(input_shape=(48, 48, 1), name="conv2d_input"),
    Conv2D(32, (3, 3), activation='relu', name="conv2d"),
    Conv2D(64, (3, 3), activation='relu', name="conv2d_1"),
    MaxPooling2D(pool_size=(2, 2), name="max_pooling2d"),
    Dropout(0.25, name="dropout"),
    Conv2D(128, (3, 3), activation='relu', name="conv2d_2"),
    MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1"),
    Conv2D(128, (3, 3), activation='relu', name="conv2d_3"),
    MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2"),
    Dropout(0.25, name="dropout_1"),
    Flatten(name="flatten"),
    Dense(1024, activation='relu', name="dense"),
    Dropout(0.5, name="dropout_2"),
    Dense(7, activation='softmax', name="dense_1")
])


# output labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# # load json and create model
# json_file = open('C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/model/emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# try:
#     emotion_model = tf.keras.models.model_from_json(loaded_model_json)
#     print("Model loaded successfully.")
# except TypeError as e:
#     print(f"An error occurred while deserializing the model: {e}")
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights('C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/model/emotion_model.h5')
print("Loaded model from disk")


# You can either take your live camera feed or paste the path of the video by commenting the other.
# start the webcam feed
cap = cv2.VideoCapture(0)

# paste your video path here 
# cap = cv2.VideoCapture("Paste path here")

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # display frame by frame
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #exit by pressing q key
        break

# release all resouces on exit
cap.release()
cv2.destroyAllWindows()

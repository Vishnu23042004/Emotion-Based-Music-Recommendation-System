import numpy as np
import cv2
from PIL import Image
from pandastable import Table, TableModel
import datetime
from threading import Thread
import time
import pandas as pd
from keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential

face_cascade=cv2.CascadeClassifier('C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/haarcascades/haarcascade_frontalface_default.xml')
ds_factor=0.6


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
# json_file = open('model//emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)

# load weights into new modele
emotion_model.load_weights('C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/model/emotion_model.h5')
print("Loaded model from disk")

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
music_dist={0:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/angry.csv",1:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/disgusted.csv",2:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/fearful.csv",3:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/happy.csv",4:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/neutral.csv",5:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/sad.csv",6:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/surprised.csv"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text=[0]


''' Class for calculating FPS while streaming. Used this to check performance of using another thread for video streaming '''
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()


''' Class for using another thread for video streaming to boost performance '''
class WebcamVideoStream:
    	
		def __init__(self, src=0):
			self.stream = cv2.VideoCapture(src,cv2.CAP_DSHOW)
			(self.grabbed, self.frame) = self.stream.read()
			self.stopped = False

		def start(self):
				# start the thread to read frames from the video stream
			Thread(target=self.update, args=()).start()
			return self
			
		def update(self):
			# keep looping infinitely until the thread is stopped
			while True:
				# if the thread indicator variable is set, stop the thread
				if self.stopped:
					return
				# otherwise, read the next frame from the stream
				(self.grabbed, self.frame) = self.stream.read()

		def read(self):
			# return the frame most recently read
			return self.frame
		def stop(self):
			# indicate that the thread should be stopped
			self.stopped = True

''' Class for reading video stream, generating prediction and recommendations '''
class VideoCamera(object):
    def __init__(self):
        self.cap1 = WebcamVideoStream(src=0).start()

    def get_frame(self):
        global df1
        image = self.cap1.read()
        image = cv2.resize(image, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        df1 = music_rec()  # Ensure df1 is updated here

        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
            roi_gray_frame = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            show_text[0] = maxindex
            cv2.putText(image, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            df1 = music_rec()

        last_frame1 = image.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(last_frame1)
        img = np.array(img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes(), df1


def music_rec():
    print(f'Fetching recommendations for: {music_dist[show_text[0]]}')
    df = pd.read_csv(music_dist[show_text[0]])
    df = df[['Name','Album','Artist']]
    df = df.head(15)
    return df


# import cv2
# import numpy as np
# from PIL import Image
# from threading import Thread
# import pandas as pd
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# # Load the face cascade
# face_cascade=cv2.CascadeClassifier('C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/haarcascades/haarcascade_frontalface_default.xml')

# # Load the emotion model
# emotion_model = Sequential([
#     InputLayer(input_shape=(48, 48, 1), name="conv2d_input"),
#     Conv2D(32, (3, 3), activation='relu', name="conv2d"),
#     Conv2D(64, (3, 3), activation='relu', name="conv2d_1"),
#     MaxPooling2D(pool_size=(2, 2), name="max_pooling2d"),
#     Dropout(0.25, name="dropout"),
#     Conv2D(128, (3, 3), activation='relu', name="conv2d_2"),
#     MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1"),
#     Conv2D(128, (3, 3), activation='relu', name="conv2d_3"),
#     MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2"),
#     Dropout(0.25, name="dropout_1"),
#     Flatten(name="flatten"),
#     Dense(1024, activation='relu', name="dense"),
#     Dropout(0.5, name="dropout_2"),
#     Dense(7, activation='softmax', name="dense_1")
# ])

# emotion_model.load_weights('C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/model/emotion_model.h5')
# print("Loaded model from disk")

# # Emotion and music dictionaries
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# music_dist={0:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/angry.csv",1:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/disgusted.csv",2:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/fearful.csv",3:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/happy.csv",4:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/neutral.csv",5:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/sad.csv",6:"C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/songs/surprised.csv"}

# global last_frame1                                    
# last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
# global show_text
# show_text = [0]

# class VideoCamera(object):
#     def __init__(self):
#         self.cap1 = WebcamVideoStream(src=0).start()

#     def get_frame(self):
#         global show_text
#         image = self.cap1.read()
#         image = cv2.resize(image, (600, 500))
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        
#         df1 = music_rec()  # Update recommendations

#         for (x, y, w, h) in face_rects:
#             cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
#             roi_gray_frame = gray[y:y + h, x:x + w]
#             cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
#             prediction = emotion_model.predict(cropped_img)
#             maxindex = int(np.argmax(prediction))
#             show_text[0] = maxindex  # Update the global emotion index
#             cv2.putText(image, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
#             df1 = music_rec()  # Update df1 based on the new emotion

#         last_frame1 = image.copy()
#         pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(pic)
#         img = np.array(img)
#         ret, jpeg = cv2.imencode('.jpg', img)
#         return jpeg.tobytes(), df1

# def music_rec():
#     global show_text
#     emotion_index = show_text[0]
#     print(f'Fetching recommendations for: {emotion_dict[emotion_index]}')
#     df = pd.read_csv(music_dist[emotion_index])
#     df = df[['Name', 'Album', 'Artist']]
#     return df.head(15)

# class WebcamVideoStream:
#     def __init__(self, src=0):
#         self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
#         (self.grabbed, self.frame) = self.stream.read()
#         self.stopped = False

#     def start(self):
#         Thread(target=self.update, args=()).start()
#         return self

#     def update(self):
#         while True:
#             if self.stopped:
#                 return
#             (self.grabbed, self.frame) = self.stream.read()

#     def read(self):
#         return self.frame

#     def stop(self):
#         self.stopped = True

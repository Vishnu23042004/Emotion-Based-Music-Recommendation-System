# import required packages
import sys
import io
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# output labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# # load json and create model
# json_file = open('model/emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)
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

# load weights into new model
emotion_model.load_weights('C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/model/emotion_model.h5')
print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
        'C:/Users/Vishnu Vardhan/Downloads/Recommendation-system-main/Recommendation-system-main/data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')




# doing prediction on test data

predictions = np.array([])
labels =  np.array([])

I = 0
for x, y in test_generator:
  predictions = np.concatenate([predictions, np.argmax(emotion_model.predict(x),axis=-1)])
  labels = np.concatenate([labels, np.argmax(y,axis=-1)])
  I += 1
  if I > 100:  # this if-break statement reduces the running time.
    break       
ConfusionMatrixDisplay(
    confusion_matrix=tf.math.confusion_matrix(
        labels=labels, predictions=predictions)
    .numpy(), display_labels=emotion_dict).plot(cmap=plt.cm.Blues)
plt.show()
print(tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy())

print("-----------------------------")
print(classification_report(labels, predictions))

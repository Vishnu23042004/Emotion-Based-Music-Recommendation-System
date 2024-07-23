# Music Recommendation-system using emotion detection
* Dataset is downloaded from https://www.kaggle.com/msambare/fer2013 
* In the same folder, create a new folder named data and save the test and train folders in it. (This repository already has the dataset download and saved in it).



### Packages need to be installed
- Run <code>pip install -r requirements.txt</code> to install all dependencies.


### To train Emotion detector model
- Run <code>Train.py</code>
- After Training , you will find the trained model structure and weights are stored in your project directory. emotion_model.json and emotion_model.h5.
- Copy these two files, create model folder in your project directory and paste it. (The pre-trained model is available in the model folder in this repository).

### To run your emotion detection file
- Run <code>detect_emotion.py</code>
- You can either take your live camera feed or paste the path of the video by commenting the code other out in Test.py 

### To analyse the model
- Run <code>Evaluate.py</code>

### To run the webapp
Run <code>app.py</code> and visit the link http://127.0.0.1:5000/ 

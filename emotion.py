import os
import sys
import cv2
import numpy as np
from PIL import Image
from datetime import datetime as dt
from keras.models import model_from_json
from keras.preprocessing import image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # to ignore initialization of GPU

WINDOWNAME = 'Face Covering with Emotion Detection'
WIDTH, HEIGHT = 1000, 700

class Emotions:
    def __init__(self, l):
        """ Constructor of the class Emotions - Called when creating the object
        Loads the tensorflow model, model weights and haarcascade, occupy the input resource i.e the default camera
        Set up configurations of indicators, recording and emoji style """

        self.model = model_from_json(open("Models/fer.json", 'r').read())
        self.model.load_weights('Models/fer.h5')
        self.lookup = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        self.haar_cascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)

        self.emo_list = ['blur', 'normal', 'cartoon', 'anime', 'food', 'traffic']
        self.emo_style = l[0] % 6
        self.ind1 = l[1]
        self.ind2 = l[2]

        self.record = False # recording is turned off by default
        self.video_writer = False
    
    def run(self):
        """ The main application loop of the software 
        Controls all the functionality of the application and calls the respective methods """

        while True:
            ret, self.img = self.cap.read()
            if not ret or np.shape(self.img) == ():
                continue

            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  
            faces_detected = self.haar_cascade.detectMultiScale(gray, 1.32, 5) # detect faces using haar cascade
            if len(faces_detected):
                pred, conf = self.predict(faces_detected, gray) # predicting emotion of the face if detected
                emo = self.lookup[pred]
            else:
                emo = 'NULL'
                conf = [0] * 7
            
            self.place_indicators(emo, conf)
            
            self.img = cv2.resize(self.img, (WIDTH,HEIGHT))
            if self.record:
                self.video_writer.write(self.img) # recording the video if toggled by user
                self.place_record()
            cv2.imshow(WINDOWNAME, self.img)

            key = cv2.waitKey(10)
            if (cv2.getWindowProperty(WINDOWNAME, cv2.WND_PROP_VISIBLE) < 1) or key == ord('q'): # exit on 'q' or cross or if window isn't visibe
                break
            elif key == ord('c'): # change emoji style on 'c'
                self.emo_style = (self.emo_style + 1) % 6
            elif key == ord('n'): # toggle confidence indicator on 'n'
                self.ind1 = (self.ind1 + 1) % 2
            elif key == ord('m'): # toggle emotion indicator on 'm'
                self.ind2 = (self.ind2 + 1) % 2
            elif key == ord('r'): # toggle record on 'r'
                if not self.video_writer:
                    self.init_recorder()
                self.record = True if not self.record else False
    
    def predict(self, faces, gray):
        """ Converts the image to a numpy array and predicts the emotion by passing the array to the tensorflow model for emotion recognition 
        The functions are trimming the image in order to extract the roi, predicting the emotion, call to function for placing emoji over the roi """

        for (x,y,w,h) in faces:  
            roi = gray[y:y+w, x:x+h]
            roi = cv2.resize(roi,(48,48))  
            img_arr = np.expand_dims(image.img_to_array(roi), axis = 0) / 255
            
            predictions = self.model.predict(img_arr)  
            pred = np.argmax(predictions[0])
            self.place_emoji(pred, x, y, w, h)

        return pred, predictions[0]   

    def overlay_image_alpha(self, img, img_overlay, x, y, alpha_mask):
        """ Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`
        `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1] """

        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
        alpha_inv = 1.0 - alpha

        img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    
    def init_recorder(self):
        """ Initialize the output file in 'Videos/' folder and creates a folder named 'Videos/' if it doesn't exist
        The output file has a name associated with the time stamp viz. `yyyymmdd_hhmmss` """

        if not os.path.exists('Videos/'):
            os.makedirs('Videos/')

        time = str(dt.now())[:19]
        time = time.replace(' ', '_')
        time = time.replace(':', '')
        time = time.replace('-', '')
        name = time + '.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mpeg')
        self.video_writer = cv2.VideoWriter()
        self.video_writer.open(os.path.join('Videos', name), fourcc, 10, (WIDTH,HEIGHT), True)

    def place_indicators(self, emo, conf):
        """ Overlay the indicators on top of the image based on the indicator flag `ind1` and `ind2` """

        if self.ind2: # place decision indicator if set to 'ON'
            cv2.putText(self.img, emo, (500,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if self.ind1: # place confidence indicator if set to 'ON'
            self.place_predbar(conf)

    def place_emoji(self, pred, x, y, w, h):
        """ Overlay the emojis on the image of the user in order to cover his face
        Reads the emoji based on the style selected (blur in case of 0), converts the image to PIL Image and calls the function `overlay_image_alpha` """

        if self.emo_style:
            img_cvt = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            img_pil = np.array(Image.fromarray(img_cvt))

            emoji = np.array(Image.open('Emojis/' + self.emo_list[self.emo_style] + '/' + self.lookup[pred] +'.png').resize((w+40, h+40)))
            if emoji.shape == ():
                return

            alpha_mask = emoji[:, :, 3] / 255.0
            img_result = img_pil[:, :, :3].copy()
            img_overlay = emoji[:, :, :3]
            self.overlay_image_alpha(img_result, img_overlay, x-20, y-20, alpha_mask)

            self.img = cv2.cvtColor(np.array(img_result), cv2.COLOR_RGB2BGR)
        else:
            self.img[y-20:y+h+20, x-20:x+w+20] = cv2.GaussianBlur(self.img[y-20:y+h+20, x-20:x+w+20], (121,121), 0)

    def place_predbar(self, predictions):
        """ Draws the prediction indicator on the image based on the list `predictions` containing prediction confidence for each emotion
        The confidence values are in the range [0,1] where 0 represents the model is sure that the emotion is not assiciated and 1 being sure about emotion """

        for i in range(7):
            cv2.putText(self.img, self.lookup[i], (10,20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.rectangle(self.img, (80, 10+(20*i)), (180, 20*(i+1)), (0, 255, 0), 2)
            cv2.rectangle(self.img, (80, 10+(20*i)), (int(80 + predictions[i]*100), 20*(i+1)), (0, 255, 0), -1)
    
    def place_record(self):
        """ Drawing the recording indicator and text onto the screen when recording is turned on """

        self.img = cv2.circle(self.img, (500,50), 10, (0,0,255), -1)
        cv2.putText(self.img, 'REC', (520,55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    def __str__(self):
        return """
        Emotions is the class which manages the main functionality and application of the software
        Syntax: object = Emotions(l)
        - `l` -> list of length 3, `l[0]` -> emoji style, `l[1]` -> confidence indicator and `l[2]` -> for emotion indicator 
        Functions:
        - Capturing the image
        - Indetifying faces and emotions
        - Placing emojis and indicators
        - Recording the video
        """

    def __del__(self):
        """ Destructor for the class Emotions
        Destroys the window and release the occupied camera resources """

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    l = list(map(int, sys.argv[1:])) if len(sys.argv) > 1 else [0,0,0]
    o = Emotions(l)
    o.run()
    del o

if __name__ == '__main__':
    main()
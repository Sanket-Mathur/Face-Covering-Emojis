import os
import sys
import cv2
import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing import image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # to ignore initialization of GPU
WINDOWNAME = 'Face Covering with Emotion Detection'

class Emotions:
    def __init__(self, l):
        """ loading the models, weights and haarcascades; occupying the input resource i.e the default camera and some other configurations """
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
        """ the main application loop of the software """
        while True:
            ret, self.img = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  
            faces_detected = self.haar_cascade.detectMultiScale(gray, 1.32, 5) # detect faces using haar cascade
            if len(faces_detected):
                pred, conf = self.predict(faces_detected, gray)
                emo = self.lookup[pred]
            else:
                emo = 'NULL'
                conf = [0] * 7
            
            self.place_indicators(emo, conf)
            
            self.img = cv2.resize(self.img, (1000,700))
            if self.record:
                self.video_writer.write(self.img)
                self.place_record()
            cv2.imshow(WINDOWNAME, self.img)

            key = cv2.waitKey(10)
            if (cv2.getWindowProperty(WINDOWNAME, cv2.WND_PROP_VISIBLE) < 1) or key == ord('q'): # exit on 'q' or cross or if window isn't visibe
                break
            elif key == ord('c'): # change style on 'c'
                self.emo_style = (self.emo_style + 1) % 6
            elif key == ord('n'): # confidence indicator on 'n'
                self.ind1 = (self.ind1 + 1) % 2
            elif key == ord('m'): # decision indicator on 'm'
                self.ind2 = (self.ind2 + 1) % 2
            elif key == ord('r'):
                if not self.video_writer:
                    self.init_recorder()
                self.record = True if not self.record else False
    
    def predict(self, faces, gray):
        """ Converts the image to an array and predicts the emotion """
        for (x,y,w,h) in faces:  
            roi = gray[y:y+w, x:x+h]
            roi = cv2.resize(roi,(48,48))  
            img_arr = np.expand_dims(image.img_to_array(roi), axis = 0) / 255
            
            predictions = self.model.predict(img_arr)  
            pred = np.argmax(predictions[0])
            self.place_emoji(pred, x, y, w, h)
        return pred, predictions[0]   

    def overlay_image_alpha(self, img, img_overlay, x, y, alpha_mask):
        """ Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`. `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1] """
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
        """ initialize the output file in Videos folder and make sure that a Videos folder exists before doing so"""
        if not os.path.exists('Videos/'):
            os.makedirs('Videos/')
        fourcc = cv2.VideoWriter_fourcc(*'mpeg')
        self.video_writer = cv2.VideoWriter()
        self.video_writer.open(os.path.join('Videos','output.mp4'), fourcc, 10, (1000,700), True)
    
    def place_indicators(self, emo, conf):
        """ placing the indicators depending on their status """
        if self.ind2: # place decision indicator if set to 'ON'
            cv2.putText(self.img, emo, (500,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if self.ind1: # place confidence indicator if set to 'ON'
            self.place_predbar(conf)

    def place_emoji(self, pred, x, y, w, h):
        """ overlaying the emoji on the image of the user in order to cover his identity """
        if self.emo_style:
            img_cvt = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            img_pil = np.array(Image.fromarray(img_cvt))

            emoji = np.array(Image.open('Emojis/' + self.emo_list[self.emo_style] + '/' + self.lookup[pred] +'.png').resize((w+40, h+40)))

            alpha_mask = emoji[:, :, 3] / 255.0
            img_result = img_pil[:, :, :3].copy()
            img_overlay = emoji[:, :, :3]
            self.overlay_image_alpha(img_result, img_overlay, x-20, y-20, alpha_mask)

            self.img = cv2.cvtColor(np.array(img_result), cv2.COLOR_RGB2BGR)
        else:
            self.img[y-20:y+h+20, x-20:x+w+20] = cv2.GaussianBlur(self.img[y-20:y+h+20, x-20:x+w+20], (121,121), 0)

    def place_predbar(self, predictions):
        """ drawing the bar representing the prediction confidence given for each from """
        for i in range(7):
            cv2.putText(self.img, self.lookup[i], (10,20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.rectangle(self.img, (80, 10+(20*i)), (180, 20*(i+1)), (0, 255, 0), 2)
            cv2.rectangle(self.img, (80, 10+(20*i)), (int(80 + predictions[i]*100), 20*(i+1)), (0, 255, 0), -1)
    
    def place_record(self):
        """ drawing the recording symbol and text onto the screen when recording is turned on """
        self.img = cv2.circle(self.img, (500,50), 10, (0,0,255), -1)
        cv2.putText(self.img, 'REC', (520,55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    def __del__(self):
        """ destroying the window and releasing the occupied resources """
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    print(sys.argv)
    l = list(map(int, sys.argv[1:])) if len(sys.argv) > 1 else [0,0,0]
    o = Emotions(l)
    o.run()
    del o

if __name__ == '__main__':
    main()
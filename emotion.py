import os
import sys
import cv2
import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing import image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # to ignore initialization of GPU

class Emotions:
    def __init__(self, l):
        """ loading the models, weights and haarcascades; occupying the input resource i.e the default camera and some other configurations """
        self.model = model_from_json(open("Models/fer.json", 'r').read())
        self.model.load_weights('Models/fer.h5')
        self.lookup = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        self.haar_cascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)

        self.emo_list = ['blur', 'normal', 'cartoon', 'anime']
        self.emo_style = l[0] % 4
        self.ind1 = l[1]
        self.ind2 = l[2]
    
    def run(self):
        """ the main application loop of the software """
        while True:
            ret, self.img = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  
            faces_detected = self.haar_cascade.detectMultiScale(gray, 1.32, 5)

            if len(faces_detected):
                pred = self.predict(faces_detected, gray)
                if self.ind2:
                    cv2.putText(self.img, self.lookup[pred], (500,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                if self.ind2:
                    cv2.putText(self.img, 'NULL', (500,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if self.ind1:
                    self.place_predbar([0,0,0,0,0,0,0])
            
            self.img = cv2.resize(self.img, (1000,700))
            cv2.imshow('Face Covering with Emotion Detection', self.img)

            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.emo_style = (self.emo_style+1) % 4
            elif key == ord('n'):
                self.ind1 = 0 if self.ind1 else 1
            elif key == ord('m'):
                self.ind2 = 0 if self.ind2 else 1
    
    def predict(self, faces, gray):
        """ Converts the image to an array and predicts the emotion """
        for (x,y,w,h) in faces:  
            roi = gray[y:y+w, x:x+h]
            roi = cv2.resize(roi,(48,48))  
            img_arr = np.expand_dims(image.img_to_array(roi), axis = 0) / 255
            
            predictions = self.model.predict(img_arr)  
            pred = np.argmax(predictions[0])
            self.place_emoji(pred, x, y, w, h)
            if self.ind1:
                self.place_predbar(predictions[0])
        return pred        

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
            self.img[y-20:y+h+20, x-20:x+w+20] = cv2.GaussianBlur(self.img[y-20:y+h+20, x-20:x+w+20], (151,151), 0)

    def place_predbar(self, predictions):
        """ drawing the bar representing the prediction confidence given for each from """
        for i in range(7):
            cv2.putText(self.img, self.lookup[i], (10,20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.rectangle(self.img, (80, 10+(20*i)), (180, 20*(i+1)), (0, 255, 0), 2)
            cv2.rectangle(self.img, (80, 10+(20*i)), (int(80 + predictions[i]*100), 20*(i+1)), (0, 255, 0), -1)

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
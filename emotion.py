import os  
import cv2  
import numpy as np  
from keras.models import model_from_json  
from keras.preprocessing import image  

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # to ignore initialization of GPU

class Emotions:
    def __init__(self):
        """ loading the models, weights and haarcascades and also occupying the input resource i.e the default camera """
        self.model = model_from_json(open("Models/fer.json", 'r').read())
        self.model.load_weights('Models/fer.h5')
        self.lookup = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        self.haar_cascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)
    
    def run(self):
        """ the main application loop of the software """
        while True:
            ret, img = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            faces_detected = self.haar_cascade.detectMultiScale(gray, 1.32, 5)

            if len(faces_detected):
                pred = self.predict(faces_detected, img, gray)
                cv2.putText(img, self.lookup[pred], (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.imshow('Face Covering with Emotion Detection', img)

            if cv2.waitKey(10) == ord('q'):
                break
    
    def predict(self, faces, img, gray):
        """ Converts the image to an array and predicts the emotion """
        for (x,y,w,h) in faces:  
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), thickness=5)  
            roi = gray[y:y+w, x:x+h]
            roi = cv2.resize(roi,(48,48))  
            img_arr = np.expand_dims(image.img_to_array(roi), axis = 0) / 255
            
            predictions = self.model.predict(img_arr)  
            pred = np.argmax(predictions[0])
            return pred            

    def __del__(self):
        """ destroying the window and releasing the occupied resources """
        self.cap.release()

def main():
    o = Emotions()
    o.run()
    del o

if __name__ == '__main__':
    main()
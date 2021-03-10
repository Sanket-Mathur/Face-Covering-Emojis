import os  
import cv2  
import numpy as np  
from PIL import Image
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
            ret, self.img = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  
            faces_detected = self.haar_cascade.detectMultiScale(gray, 1.32, 5)

            if len(faces_detected):
                pred = self.predict(faces_detected, gray)
                cv2.putText(self.img, self.lookup[pred], (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.imshow('Face Covering with Emotion Detection', self.img)

            if cv2.waitKey(10) == ord('q'):
                break
    
    def predict(self, faces, gray):
        """ Converts the image to an array and predicts the emotion """
        for (x,y,w,h) in faces:  
            roi = gray[y:y+w, x:x+h]
            roi = cv2.resize(roi,(48,48))  
            img_arr = np.expand_dims(image.img_to_array(roi), axis = 0) / 255
            
            predictions = self.model.predict(img_arr)  
            pred = np.argmax(predictions[0])
            self.place_emoji(pred, x, y, w, h)
        return pred        

    def overlay_image_alpha(self, img, img_overlay, x, y, alpha_mask):
        """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.
        `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1]."""
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        # Blend overlay within the determined ranges
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
        alpha_inv = 1.0 - alpha

        img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

    def place_emoji(self, pred, x, y, w, h):
        img_cvt = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img_pil = np.array(Image.fromarray(img_cvt))

        emoji = np.array(Image.open('Emojis/normal/' + self.lookup[pred] +'.png').resize((w+40, h+40)))

        alpha_mask = emoji[:, :, 3] / 255.0
        img_result = img_pil[:, :, :3].copy()
        img_overlay = emoji[:, :, :3]
        self.overlay_image_alpha(img_result, img_overlay, x-20, y-20, alpha_mask)

        self.img = cv2.cvtColor(np.array(img_result), cv2.COLOR_RGB2BGR)

    def __del__(self):
        """ destroying the window and releasing the occupied resources """
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    o = Emotions()
    o.run()
    del o

if __name__ == '__main__':
    main()
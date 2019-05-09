
# Detects whether it is human or dog or something else

import cv2
import numpy as np
from CNNmodels.util import path_to_tensor, paths_to_tensor
from CNNmodels.dog_breed_detection import DogBreedClassifier
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf


class DogDetector:
    def __init__(self):
        self.graph = tf.get_default_graph()
        #self.img_path = img_path
        #pass

    # returns "True" if the human face is detected in image stored at img_path
    def human_face_detector(self):
        face_cascade = cv2.CascadeClassifier('harcascade/haarcascade_frontalface_alt.xml')
        img = cv2.imread(self.img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    # We will use a pre-trained ResNet-50 model to detect dogs in images. The ResNet-50 model is trained on
    # ImageNet data-set
    def resnet50_predict_labels(self):
        # returns prediction vector for image located at img_path
        ResNet50_model = ResNet50(weights='imagenet')
        img = preprocess_input(path_to_tensor(self.img_path))
        return np.argmax(ResNet50_model.predict(img))

    # returns "True" if a dog is detected in the image stored at img_path
    # The dog label values are in between 151 and 268 (inclusive) in ImageNet data-set
    def dog_detector(self):
        # define ResNet50 model
        prediction = self.resnet50_predict_labels()
        return (prediction <= 268) & (prediction >= 151)

    def image_prediction(self, img_path):
        self.img_path = img_path
        with self.graph.as_default():
            if self.dog_detector():
                print("It's a dog!")
                dog = DogBreedClassifier(self.img_path)
                result = dog.Xception_predict_breed()
                print("And I guess the Breed of this dog is a {}".format(result))
                return 3, result

            elif self.human_face_detector():
                print("It's a human!")
                return 1, {'prob': [1], 'breeds': ["Its a human"]}

            else:
                print("This looks neither a dog, nor human; must be something else .")
                return 2, {'prob': [1], 'breeds': ["neither a dog, nor human"]}


if __name__ == "__main__":
    dog = DogDetector()
    dog.image_prediction("data/train/0a4f1e17d720cdff35814651402b7cf4.jpg")

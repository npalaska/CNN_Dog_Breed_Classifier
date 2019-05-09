
# Detects which dog breed is the given image

import numpy as np
from glob import glob
from CNNmodels.util import path_to_tensor, extract_Xception
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.models import Sequential

class DogBreedClassifier:
    def __init__(self, img_path):
        self.img_path = img_path

    def data(self):
        bottleneck_features = np.load('/data/bottleneck_features/DogXceptionData.npz')
        train_Xception = bottleneck_features['train']
        valid_Xception = bottleneck_features['valid']
        test_Xception = bottleneck_features['test']


    def network_def(self):
        Xception_model = Sequential()
        Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
        Xception_model.add(Dense(500, activation='relu'))
        Xception_model.add(Dropout(0.4))

        Xception_model.add(Dense(500, activation='relu'))
        Xception_model.add(Dropout(0.4))

        Xception_model.add(Dense(133, activation='softmax'))
        return Xception_model

    def compile_model(self):
        xception_model = self.network_def()
        xception_model.load_weights('saved_models/weights.best.Xception.hdf5')
        return xception_model

    def Xception_predict_breed(self):
        dog_names = [item[20:-1] for item in sorted(glob("data/dogImages/train/*/"))]
        print(len(dog_names))
        print(dog_names)
        xception_model = self.compile_model()
        bottleneck_feature = extract_Xception(path_to_tensor(self.img_path))  # extract bottleneck features
        predicted_vector = xception_model.predict(bottleneck_feature)  # obtain predicted vector
        top_5 = predicted_vector[0].argsort()[-5:][::-1]
        print(predicted_vector[0].argsort()[-5:][::-1])
        print(np.argmax(predicted_vector))
        print([predicted_vector[0][i] for i in top_5])
        prob = [predicted_vector[0][i] for i in top_5]
        breeds = [dog_names[i] for i in top_5]
        result = {'prob': prob, 'breeds': [dog_names[i] for i in top_5]}
        #return dog_names[np.argmax(predicted_vector)]
        return result


if __name__ == "__main__":
    dog = DogBreedClassifier("static/img/ref/Akita.jpg")
    print(dog.Xception_predict_breed())

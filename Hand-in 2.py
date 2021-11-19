import requests
import json
import cv2
import numpy as np
from PIL import Image
from keras import models
import tensorflow as tf

model = models.load_model('keras_model.h5')
video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()
        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into dimensions you used while training
        im = im.resize((224,224))
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = (img_array.astype(np.float32) / 127.0) - 1

        #Calling the predict function using keras
        prediction = model.predict(img_array)
        labels = ['Apple', 'Banana', 'Beetroot', 'Bell pepper', 'Cabbage', 'Carrot', 'Cauliflower', 'Chili','Class 9', 'Cucumber', 'Eggplant', 'Garlic', 'Ginger', 'Grape', 'Jalapeno', 'Kiwi', 'Lemon', 'Lettuce', 'Mango',
        'Onion', 'Orange', 'Paprika', 'Pear', 'Pea', 'Pineapple', 'Pomegranate', 'Potato', 'Raddish', 'Soy bean', 'Spinach', 'Sweetpotato', 'Tomato', 'Turnip', 'Watermelon']

        print(labels[np.argmax(prediction)])

        cv2.imshow("Prediction", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
      # if key == ord('p'):
       #         break

video.release()
cv2.destroyAllWindows()

url = "https://api.edamam.com/api/recipes/v2?type=public&q=tomato&app_id=6270b8ac&app_key=ca87ddf9ee7a96725492d03875534c30"
apiKey = "ca87ddf9ee7a96725492d03875534c30"
appID = "6270b8ac"

r = requests.get(url)
r.status_code
r.encoding
r.text
recipes = r.json()


print(recipes['hits'][0]['recipe']['ingredientLines'][0])
print(recipes['hits'][0]['recipe']['ingredientLines'][1])



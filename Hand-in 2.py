import requests
import cv2
import numpy as np
from PIL import Image
from keras import models

apiKey = "ca87ddf9ee7a96725492d03875534c30"
appID = "6270b8ac"

model = models.load_model('keras_model.h5')
video = cv2.VideoCapture(0)

ingridients = []

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
        label = labels[np.argmax(prediction)]

        cv2.imshow("Prediction", frame)

        key=cv2.waitKey(1)
        if key == ord('q'):
                break

        if key == ord('p'):
                ingridient = labels[np.argmax(prediction)]+"%20 "
                ingridients.append(ingridient)


video.release()
cv2.destroyAllWindows()
finalingredients = ""
for x in ingridients:
        finalingredients+=x
print(finalingredients)
# HTTP request
url = "https://api.edamam.com/api/recipes/v2?"
headers = {'type': 'public', 'q': finalingredients, 'app_id': appID, 'app_key': apiKey}

r = requests.get(url, params=headers)
recipes = r.json()

# for loop that iterates through the Json document and prints out the first 5 recipes and their ingredients list
for x in range(0,5):
        try:
                ingredientLines = recipes['hits'][x]['recipe']['ingredientLines']
                print("recipe ",x+1,":")
                print(recipes['hits'][x]['recipe']['label'])
                print("\nIngredients:")
                for ingredients in ingredientLines:
                        print(ingredients)
                print("\n")
        except:
                print("No more recipes!")
                break



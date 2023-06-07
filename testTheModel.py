from keras.models import load_model
import numpy as np
from keras.utils import load_img, img_to_array, save_img
from os import walk
import tensorflow as tf
from tqdm import tqdm

f = []
DIR_PATH = "testData/TGP/"

for (dirpath, dirnames, filenames) in walk(DIR_PATH):
    f.extend(filenames)
    break

# load the model
loaded_model = load_model("")

class_names = ['Nothing', 'The Good Place', 'Avatar']
count_map = {'Avatar': 0, 'The Good Place': 0, 'Nothing': 0}
file_name_map = {'Avatar': [], 'The Good Place': [], 'Nothing': []}

# Load and preprocess the image
for file in tqdm(f):
    img_path = DIR_PATH
    image = load_img(img_path+file)
    image = tf.image.resize(image, [224, 224])
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.copy(image_array)
    image_array /= 255.

    # Make predictions on the image
    predictions = loaded_model.predict(image_array, verbose=0)
    # Print the predicted class name and confidence level
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    count_map[predicted_class_name] += 1
    file_name_map[predicted_class_name].append(file)

print(file_name_map)
print(count_map)

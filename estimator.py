# Import requirements
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

# Unique breeds of dogs
labels = pd.read_csv("./data/labels.csv")["breed"].to_numpy()
unique_breeds = np.unique(labels)


def load_model(model_path):
  """
  Loads a saved model from specific path
  """
  print(f'Loading saved model from: {model_path}')
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model
  

# Define image size
IMG_SIZE = 224
# Create a function for preprocessing images
def process_image(image_path, img_size=IMG_SIZE):
  """
  Takes an image file path and turns the image into a Tensor.
  """
  # Read an image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numeric tensor with 3 color channels (Red, Gree, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the color channel values from 0-255 to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desire value(244, 244)
  image = tf.image.resize(image, size=[img_size, img_size])
  
  return image


# Batch passed image
BATCH_SIZE = 32 # Its default value is 32
# Create a function to turn data into batches -> test
def create_data_batch(X, y=None, batch_size=BATCH_SIZE):
  """
  Create batches of data from image (X) and label (y).
  """
  # if data is a test dataset, we don't have labels
  print("Creating test data batches...")
  data = tf.data.Dataset.from_tensor_slices((tf.constant(tf.reshape(X, [-1]))))
  data_batch = data.map(process_image).batch(BATCH_SIZE)

  return data_batch


# Get the predicted label
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilites into a label
  """
  return unique_breeds[np.argmax(prediction_probabilities)]


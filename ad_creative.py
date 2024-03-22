
!pip install keras_cv

import tensorflow as tf
from keras_cv.models import StableDiffusion
import matplotlib.pyplot as plt
import numpy as np

# Initialize the Stable Diffusion model
model_diffusion = StableDiffusion(img_width=512, img_height=512)

# Load a pre-trained MobileNetV2 model for image classification
model_classification = tf.keras.applications.MobileNetV2(weights='imagenet')

def generate_images(prompt, batch_size=3):
    # Generate images based on the prompt
    images = model_diffusion.text_to_image(prompt, batch_size=batch_size)
    return images

def plot_images(images):
    # Plot the generated images
    plt.figure(figsize=(20, 20))
    for i, image in enumerate(images):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(image)
        plt.axis("off")
    plt.show()

def preprocess_images(images):
    # Resize images to match the input shape expected by MobileNetV2 (224x224)
    resized_images = [tf.image.resize(img, (224, 224)) for img in images]
    # Preprocess images for MobileNetV2
    preprocessed_images = [tf.keras.applications.mobilenet_v2.preprocess_input(img) for img in resized_images]
    preprocessed_images = np.array(preprocessed_images)
    return preprocessed_images


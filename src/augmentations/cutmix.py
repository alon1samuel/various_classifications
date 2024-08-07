"""
Title: CutMix data augmentation for image classification
Author: [Sayan Nath](https://twitter.com/sayannath2350)
Date created: 2021/06/08
Last modified: 2023/11/14
Description: Data augmentation with CutMix for image classification on CIFAR-10.
Accelerator: GPU
Converted to Keras 3 By: [Piyush Thakur](https://github.com/cosmo3769)
"""

"""
## Introduction
"""

import numpy as np
import keras
import matplotlib.pyplot as plt

from keras import layers

# TF imports related to tf.data preprocessing
from tensorflow import clip_by_value
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import random as tf_random

keras.utils.set_random_seed(42)

"""
## Load the CIFAR-10 dataset

In this example, we will use the
[CIFAR-10 image classification dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
"""

def preprocess_image(image, label):
    image = tf_image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf_image.convert_image_dtype(image, "float32") / 255.0
    label = keras.ops.cast(label, dtype="float32")
    return image, label


"""
## Convert the data into TensorFlow `Dataset` objects
"""

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf_random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf_random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def get_box(lambda_value, img_size):
    cut_rat = keras.ops.sqrt(1.0 - lambda_value)

    cut_w = img_size * cut_rat  # rw
    cut_w = keras.ops.cast(cut_w, "int32")

    cut_h = img_size * cut_rat  # rh
    cut_h = keras.ops.cast(cut_h, "int32")

    cut_x = keras.random.uniform((1,), minval=0, maxval=img_size)  # rx
    cut_x = keras.ops.cast(cut_x, "int32")
    cut_y = keras.random.uniform((1,), minval=0, maxval=img_size)  # ry
    cut_y = keras.ops.cast(cut_y, "int32")

    boundaryx1 = clip_by_value(cut_x[0] - cut_w // 2, 0, img_size)
    boundaryy1 = clip_by_value(cut_y[0] - cut_h // 2, 0, img_size)
    bbx2 = clip_by_value(cut_x[0] + cut_w // 2, 0, img_size)
    bby2 = clip_by_value(cut_y[0] + cut_h // 2, 0, img_size)

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    return boundaryx1, boundaryy1, target_h, target_w


def cutmix(train_ds_one, train_ds_two, alpha_p=0.2):
    (image1, label1), (image2, label2) = train_ds_one, train_ds_two
    
    img_size = image1.shape[1]

    alpha = [alpha_p]
    beta = [alpha_p]

    # Get a sample from the Beta distribution
    lambda_value = sample_beta_distribution(1, alpha, beta)

    # Define Lambda
    lambda_value = lambda_value[0][0]

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value, img_size)

    # Get a patch from the second image (`image2`)
    crop2 = tf_image.crop_to_bounding_box(
        image2, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image2` patch (`crop2`) with the same offset
    image2 = tf_image.pad_to_bounding_box(
        crop2, boundaryy1, boundaryx1, img_size, img_size
    )
    # Get a patch from the first image (`image1`)
    crop1 = tf_image.crop_to_bounding_box(
        image1, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image1` patch (`crop1`) with the same offset
    img1 = tf_image.pad_to_bounding_box(
        crop1, boundaryy1, boundaryx1, img_size, img_size
    )

    # Modify the first image by subtracting the patch from `image1`
    # (before applying the `image2` patch)
    image1 = image1 - img1
    # Add the modified `image1` and `image2`  together to get the CutMix image
    image = image1 + image2

    # Adjust Lambda in accordance to the pixel ration
    lambda_value = 1 - (target_w * target_h) / (img_size * img_size)
    lambda_value = keras.ops.cast(lambda_value, "float32")

    # Combine the labels of both images
    label = lambda_value * label1 + (1 - lambda_value) * label2
    return image, label


"""
**Note**: we are combining two images to create a single one.
## Visualize the new dataset after applying the CutMix augmentation
"""

def dataset_cutmixed(train_ds, alpha):
    AUTO = tf_data.AUTOTUNE
    train_ds = tf_data.Dataset.zip((
        train_ds.shuffle(10000, seed=1, reshuffle_each_iteration=True), 
        train_ds.shuffle(10000, seed=10, reshuffle_each_iteration=True), 
        ))
    
    def cutmix_wraper(ds_1, ds_2):
        return cutmix(ds_1, ds_2, alpha)

    # Create the new dataset using our `cutmix` utility

    train_ds_cmu = (
        train_ds.shuffle(1024)
        .map(cutmix_wraper, num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )

    return train_ds_cmu




def main():

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    class_names = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]

    """
    ## Define hyperparameters
    """

    AUTO = tf_data.AUTOTUNE
    BATCH_SIZE = 32

    train_ds_one = (
        tf_data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(1024)
        .map(preprocess_image, num_parallel_calls=AUTO)
    )
    train_ds_two = (
        tf_data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(1024)
        .map(preprocess_image, num_parallel_calls=AUTO)
    )

    train_ds_simple = tf_data.Dataset.from_tensor_slices((x_train, y_train))

    test_ds = tf_data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds_simple = (
        train_ds_simple.map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    # Combine two shuffled datasets from the same training data.
    train_ds = tf_data.Dataset.zip((train_ds_one, train_ds_two))

    test_ds = (
        test_ds.map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    """
    ## Define the CutMix data augmentation function

    The CutMix function takes two `image` and `label` pairs to perform the augmentation.
    It samples `λ(l)` from the [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
    and returns a bounding box from `get_box` function. We then crop the second image (`image2`)
    and pad this image in the final padded image at the same location.
    """



    # Create the new dataset using our `cutmix` utility
    train_ds_cmu = (
        train_ds.shuffle(1024)
        .map(cutmix, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    # Let's preview 9 samples from the dataset
    image_batch, label_batch = next(iter(train_ds_cmu))
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.title(class_names[np.argmax(label_batch[i])])
        plt.imshow(image_batch[i])
        plt.axis("off")

    """
    ## Define a ResNet-20 model
    """

    """
    ## Train the model with the dataset augmented by CutMix
    """
    model = training_model()
    model.load_weights("initial_weights.weights.h5")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(train_ds_cmu, validation_data=test_ds, epochs=15)

    test_loss, test_accuracy = model.evaluate(test_ds)
    print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

    """
    ## Notes

    In this example, we trained our model for 15 epochs.
    In our experiment, the model with CutMix achieves a better accuracy on the CIFAR-10 dataset
    (77.34% in our experiment) compared to the model that doesn't use the augmentation (66.90%).
    You may notice it takes less time to train the model with the CutMix augmentation.

    You can experiment further with the CutMix technique by following the
    [original paper](https://arxiv.org/abs/1905.04899).
    """


if __name__=='__main__':
    main()
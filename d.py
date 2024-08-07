# %% [markdown]
# # Basic classification cifar 10

# %% [markdown]
# The Wandb can be viewed at - https://wandb.ai/alon_pole/cifar10 
# 
# This guide trains a neural network model to classify images of clothing, like sneakers and shirts. It's okay if you don't understand all the details; this is a fast-paced overview of a complete TensorFlow program with the details explained as you go.
# 
# This guide uses [tf.keras](https://www.tensorflow.org/guide/keras), a high-level API to build and train models in TensorFlow.

# %%
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

from src.augmentations import cutmix

# %% [markdown]
# ## Import the Cifar10 dataset

# %%
from pathlib import Path

cifar_data_path = Path('data/cifar-10/raw/tfds')
batch_size = 32
input_shape = (32, 32, 3)
IMG_SIZE = 256



# %%
# To consider to switch to - 

# If not - this is a tf.data.Dataset, docs - https://www.tensorflow.org/api_docs/python/tf/data/Dataset

#  To consider using - tf.keras.utils.image_dataset_from_directory
# example from - https://www.tensorflow.org/tutorials/load_data/images
#  https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory


train_ds = tf.keras.utils.image_dataset_from_directory(
  cifar_data_path / 'train',
  seed=123,
  image_size=input_shape[:-1],
  batch_size=batch_size
  )
test_ds = tf.keras.utils.image_dataset_from_directory(
  cifar_data_path / 'test',
  seed=123,
  image_size=input_shape[:-1],
  batch_size=batch_size
  )

# %%
class_names = np.array(sorted([item.name for item in (cifar_data_path / 'train').glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

# %%
for f in train_ds.take(1):
  print(f[0].shape)

# %%
print(tf.data.experimental.cardinality(train_ds).numpy()*32)

# %%
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
num_classes = len(class_names)

# %%

images, labels = next(iter(train_ds))
_ = plt.imshow(tf.cast(images[0], tf.uint8))
_ = plt.title(class_names[labels[0]])

# %%
import tensorflow_models as tfm

buffer_size = 10000

# implementing data augmentations like in ConvNext - Mixup, cutmix, RandAugment & RandomErasing
# papers params
mixup_param = 0.8
cutmix_param = 1.0
random_erasing = 0.25
rand_aug = (9, 0.5)

def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 255.0)
  return image, label

def prepare_ds(ds, shuffle=False, augment=False):
  # Resize and rescale all datasets.

  if shuffle:
    ds = ds.shuffle(buffer_size)

  # Use data augmentation only on the training set.
  if augment:
    ds = cutmix.dataset_cutmixed(ds, cutmix_param)

  return ds 

# %%
train_ds = prepare_ds(train_ds, shuffle=True, augment=True)
test_ds = prepare_ds(test_ds)

# %%
# To continue from - https://www.tensorflow.org/tutorials/images/data_augmentation#apply_the_preprocessing_layers_to_the_datasets

random_batch = train_ds.shuffle(buffer_size=buffer_size).take(1)


num_cols = 5

num_rows = len(images) // num_cols + (1 if len(images) % num_cols != 0 else 0)

# Create a figure with subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each image in a subplot
for img, label, ax in zip(images, labels, axes):
    ax.imshow(img, cmap='gray')  # Use cmap='gray' for grayscale images
    ax.set_title(f'Label: {label}')
    ax.axis('off')

# %%
normalization_layer = tf.keras.layers.Rescaling(1./255)

# %%
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# %%
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
num_classes=len(class_names)

# %% [markdown]
# Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255. It's important that the *training set* and the *testing set* be preprocessed in the same way:

# %% [markdown]
# ## Build the model
# 
# Building the neural network requires configuring the layers of the model, then compiling the model.

# %% [markdown]
# ### Set up the layers
# 
# The basic building block of a neural network is the [*layer*](https://www.tensorflow.org/api_docs/python/tf/keras/layers). Layers extract representations from the data fed into them. Hopefully, these representations are meaningful for the problem at hand.
# 
# Most of deep learning consists of chaining together simple layers. Most layers, such as `tf.keras.layers.Dense`, have parameters that are learned during training.

# %%
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

# %% [markdown]
# The first layer in this network, `tf.keras.layers.Flatten`, transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels). Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.
# 
# After the pixels are flattened, the network consists of a sequence of two `tf.keras.layers.Dense` layers. These are densely connected, or fully connected, neural layers. The first `Dense` layer has 128 nodes (or neurons). The second (and last) layer returns a logits array with length of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.
# 
# ### Compile the model
# 
# Before the model is ready for training, it needs a few more settings. These are added during the model's [*compile*](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile) step:
# 
# * [*Optimizer*](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) —This is how the model is updated based on the data it sees and its loss function.
# * [*Loss function*](https://www.tensorflow.org/api_docs/python/tf/keras/losses) —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
# * [*Metrics*](https://www.tensorflow.org/api_docs/python/tf/keras/metrics) —Used to monitor the training and testing steps. The following example uses *accuracy*, the fraction of the images that are correctly classified.

# %%
import math

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

lr = 4e-3
min_lr = 1e-6
warmup_epochs = 20
epochs = 300 # maybe not, we'll see. 
num_training_steps_per_epoch = len(train_ds)



lr_schedule_values = cosine_scheduler(
    lr, min_lr, epochs, num_training_steps_per_epoch,
    warmup_epochs=warmup_epochs
)

steps = [step for step in range(len(lr_schedule_values))]
print("steps amount -", steps)

plt.plot(steps, lr_schedule_values)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Warmup Oscillating Decay Learning Rate Schedule')
plt.show()


# %%
print("steps amount -", len(steps))

# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class WarmupOscillatingDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, target_learning_rate):
        super(WarmupOscillatingDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.target_learning_rate = target_learning_rate

    def __call__(self, step):
        # Warm-up phase
        step = tf.cast(step, dtype=tf.float32)
        warmup_lr = tf.cond(step < self.warmup_steps,
                            lambda: self.initial_learning_rate * step / self.warmup_steps,
                            lambda: self.initial_learning_rate)
        # Oscillating decay phase
        decay_lr = self.target_learning_rate + 0.5 * (self.initial_learning_rate - self.target_learning_rate) * (
            1 + tf.math.cos(np.pi * (step - self.warmup_steps) / self.decay_steps))
        # Combine warm-up and decay
        learning_rate = tf.cond(step < self.warmup_steps,
                                lambda: warmup_lr,
                                lambda: decay_lr)
        return learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "target_learning_rate": self.target_learning_rate
        }


plot_initial_lr = 4e-3
plot_target_lr = 1e-6
plot_warmup_epochs = 1
plot_epochs = 2 # maybe not, we'll see. 
plot_warmup_steps = plot_warmup_epochs*len(train_ds)
plot_decay_steps = plot_epochs * len(train_ds)

plot_lr_scheduler = WarmupOscillatingDecay(plot_initial_lr, plot_decay_steps, plot_warmup_steps, plot_target_lr)

# Plotting the learning rate schedule for visualization

steps = np.arange(0, plot_decay_steps + plot_warmup_steps)
learning_rates = [plot_lr_scheduler(step) for step in steps]

plt.plot(steps, learning_rates)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Warmup Oscillating Decay Learning Rate Schedule')
plt.show()

# ...


# %%
plt.plot(steps, learning_rates)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Warmup Oscillating Decay Learning Rate Schedule')
plt.yscale('log')
plt.show()


# %%
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
def categorical_crossentropy_loss(y_true, y_pred):
    y_true_int = tf.cast(y_true, tf.uint8)
    y_true_onehot = tf.one_hot(y_true_int, num_classes)
    return cce(y_true_onehot, y_pred)


# ConvNext paper params
initial_lr = 4e-3
target_lr = 1e-6
warmup_epochs = 20
epochs = 100 # maybe not, we'll see. 
warmup_steps = warmup_epochs*len(train_ds)
decay_steps = epochs * len(train_ds)

lr_scheduler = WarmupOscillatingDecay(initial_lr, decay_steps, warmup_steps, target_lr)

model.compile(optimizer=tf.keras.optimizers.AdamW(
    learning_rate=lr_scheduler,
    weight_decay=0.05,
    
),
            #   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              loss=categorical_crossentropy_loss,
              metrics=['accuracy'])

# %% [markdown]
# ## Train the model
# 
# Training the neural network model requires the following steps:
# 
# 1. Feed the training data to the model. In this example, the training data is in the `train_images` and `train_labels` arrays.
# 2. The model learns to associate images and labels.
# 3. You ask the model to make predictions about a test set—in this example, the `test_images` array.
# 4. Verify that the predictions match the labels from the `test_labels` array.
# 

# %% [markdown]
# ### Wandb init
# Initialise Wandb and set parameters

# %%
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb
wandb.login()

# %%
# Start a run, tracking hyperparameters
wandb.init(
    project="cifar10",
    name="cosine_decay___"
)

# %% [markdown]
# ### Feed the model
# 
# To start training,  call the [`model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) method—so called because it "fits" the model to the training data:

# %%
history = model.fit(
    train_ds, 
    epochs=100,
    validation_data=test_ds,
    callbacks=[WandbMetricsLogger()]
    )

# %%
f[1].numpy()

# %%
wandb.finish()

# %%
!rm -rf wandb



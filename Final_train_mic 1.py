import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras import models
from keras import regularizers
from IPython import display
import sounddevice as sd
from scipy.io.wavfile import write
import time
import tempfile
import tensorflow_model_optimization as tfmot
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

###################################################
########## Functions needed
###################################################

# Convert model to TFLITE
def convert_model_tflite(model, name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model) #creating converter object in keras
    converter.optimizations = [tf.lite.Optimize.DEFAULT] #default optimizer for tflite
    converter.target_spec.supported_types = [tf.float16] #float16 data types to shrink model size
    converted_model = converter.convert() #call convert


    tflite_file = name + ".tflite" #file name for tflite file


    with open(tflite_file, 'wb') as f:
        f.write(converted_model) #write converted model to tflite

    return tflite_file #returns the file path of the file


#removes last dimension for preprocessing
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels


# Convert raw data to a SPECTROGRAM, a 2D image that represents the frequency info
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


# Create a dataset from the spectrograms
def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label), #apply spectrogram function
      num_parallel_calls=tf.data.AUTOTUNE) 


###################################################
########## End of Functions needed
###################################################


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#define datapath variables
DATASET_PATH = 'Combo_Splits/train'
data_dir = pathlib.Path(DATASET_PATH)


# Load up full audio dataset. Requires .wav files
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64, 
    validation_split=0.2,
    seed=seed,
    #shuffle=False,
    output_sequence_length=441000, #change to match (bit rate x length in seconds). In this case, 44.1khz x 10 seconds
    subset='both') #validation and train splits

#create labels names and print
label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

# Training and val dataset need to be preprocessed with squeeze
train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# Call function to convert audio data to spectrogram
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)

#used to determine shape
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

#get shape from above.
input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))


#This model produced overall highest accuracy of around 95%
model = models.Sequential([
    layers.Input(shape=input_shape),
    #Downsample the input.
    layers.Resizing(32, 32),
    #Normalize.
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.AveragePooling2D(),  # Consider using AveragePooling2D
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5), #dropout to reduce overfitting, specifically because of small sample size
    layers.Dense(num_labels, activation='softmax'),  # Use softmax for classification
])

# model = models.Sequential([
#     layers.Input(shape=input_shape),
#     # Downsample the input.
#     layers.Resizing(32, 32),
#     # Normalize.
#     norm_layer,
#     layers.Conv2D(64, 3, activation='relu'),
#     layers.Conv2D(128, 3, activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.5),
#     layers.Flatten(),
#     layers.Dense(256, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(num_labels),
# ])

# model = models.Sequential([
#     layers.Input(shape=input_shape),
#     layers.Rescaling(1./255),  # Scale input to [0, 1] range
#     layers.Conv2D(64, 3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.25),

#     layers.Conv2D(128, 3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.25),

#     layers.Conv2D(256, 3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.25),

#     layers.Flatten(),
#     layers.Dense(512, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.5),

#     layers.Dense(num_labels)  # Using softmax for multi-class classification
# ])


model.summary() #model info displayed


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), #needed small learning rate to reduce overfitting.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'], #tracking accuracy
)

EPOCHS = 100 #generally would end early with patience=10
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
)


model.evaluate(val_spectrogram_ds, return_dict=True) #evaluate model based on the validation spectrogram

#save the models
model.save('combo3.h5', include_optimizer = True)
b_tflite = convert_model_tflite(model,"combo3")



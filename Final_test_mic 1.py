import numpy as np
import tensorflow as tf
import pathlib

#setting seed to the same value as the train.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#path to out test dataset
DATASET_PATH = 'Combo_Splits/test'
data_dir = pathlib.Path(DATASET_PATH)


##start of helper functions##
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

##end of helper functions##

# Load test data and preprocess it. requires .wav files
test_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=seed,
    output_sequence_length=441000,#change to match (bit rate x length in seconds). In this case, 44.1khz x 10 seconds
    subset='both') #use both splits in for testing

# Apply preprocessing squeeze function and then our make spectrogram function
test_ds = test_ds.map(squeeze, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = make_spec_ds(test_ds)

# Load your trained model
model = tf.keras.models.load_model('combo3.h5')

# Evaluate the model on the preprocessed test data
results = model.evaluate(test_ds, verbose=1)
print(f"Test results - Loss: {results[0]}, Accuracy: {results[1]}")
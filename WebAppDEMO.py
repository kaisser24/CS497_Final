import os
import pathlib
import numpy as np
import tensorflow as tf
import sounddevice as sd
from scipy.io.wavfile import write
import time
import scipy.signal as sps

subtype = 'PCM_16'
dtype = 'float'

#Grabs samplerate from wav files
samplerate = 44100  # may need to change based on wav files
sd.default.samplerate = samplerate
sd.default.channels = 1

label_names = ['Ariana Grande', 'Beyonce', 'Britney Spears', 'Drake', 'Michael Jackson', 'SZA', 'Taylor Swift',
               'The Weeknd']

def predict_artist():
    #similar to other demo script, except web app uses function call
    print('Commands:', label_names)
    print("capture data, send to NN")

    #loads input model
    interpreter = tf.lite.Interpreter(model_path="combo3.tflite")
    interpreter.allocate_tensors()

    # Output shape
    myrecording = sd.rec(int(10 * samplerate), dtype=dtype)
    sd.wait()

    myrecording = tf.squeeze(myrecording, axis=-1)

    x = get_spectrogram(myrecording)
    xin = tf.expand_dims(x, 0)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.float32(xin)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = label_names[np.argmax(output_data, axis=1)[0]]
    #compare web results to predicted label
    print("Predicted Artist:", predicted_label)
    return predicted_label

def get_spectrogram(waveform):
    print("**********WAVEFORM*********")
    print(waveform.shape)
    print("********** ---- WAVEFORM*********")

    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram
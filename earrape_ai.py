import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from tensorflow.keras import layers

import random
import time

from scipy.io.wavfile import read
from scipy.fft import fft

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

test_file = tf.io.read_file('data/vichrica.wav')
audio, _ = tf.audio.decode_wav(contents=test_file)

print(audio.shape)
audio_slice = audio[48000:49920].numpy().flatten()[0::2]
print(audio_slice.shape)

tensor = tf.cast(audio_slice, tf.float32) / 32768.0

# Convert to spectrogram
spectrogram = tfio.audio.spectrogram(tensor, nfft=128, window=128, stride=32)

print(spectrogram.shape)

# plt.figure()
# plt.imshow(tf.math.log(spectrogram).numpy())
# plt.show()



class DataProvider:
    def __init__(self, training_path: str) -> None:
        self.training_path: str = training_path
        self.training_files = os.listdir(self.training_path)
        self.parts_buffer: list = []

        self.index = 0

    def next(self):
        print("next")
        if len(self.parts_buffer) == 0:
            self.load_next_file()
        
        tensor = self.parts_buffer.pop().astype(np.float32) / 32768.0
        spectrogram = fft(tensor)
        spectrogram = np.abs(spectrogram)

        return spectrogram

    def get_batch(self, size: int):
        return np.array([self.next() for _ in range(size)])

    def load_next_file(self):
        print(f"Loading file {self.index}")
        file_path = os.path.join(self.training_path, self.training_files[self.index])
        
        self.index += 1
        if self.index == len(self.training_files):
            self.index = 0
            random.shuffle(self.training_files)
        
        position = 0

        self.parts_buffer = []
        sr, audio = read(file_path)

        while position < audio.shape[0] - 1:
            self.parts_buffer.append(
                audio[position: position + 1920][:, 0]
            )
            position += 1920
        
data = DataProvider("data")



# def get_spectrogram(waveform):
#     # Zero-padding for an audio waveform with less than 1920 samples.
#     input_len = 1920
#     waveform = waveform[:input_len]
#     # Cast the waveform tensors' dtype to float32.
#     waveform = tf.cast(waveform, dtype=tf.float32)
#     # Convert the waveform to a spectrogram via a STFT.
#     spectrogram = tf.signal.stft(
#         waveform, frame_length=48, frame_step=48)
#     # Obtain the magnitude of the STFT.
#     spectrogram = tf.abs(spectrogram)
#     # Add a `channels` dimension, so that the spectrogram can be used
#     # as image-like input data with convolution layers (which expect
#     # shape (`batch_size`, `height`, `width`, `channels`).
#     spectrogram = spectrogram[..., tf.newaxis]
#     return spectrogram

# test_file = tf.io.read_file("vichrica.wav")
# test_audio, _ = tf.audio.decode_wav(contents=test_file)

# print(test_audio[:1920, 0].shape)
# spectrogram = get_spectrogram(test_audio[:1920, 0])
# print(spectrogram)

norm_layer = layers.Normalization()

def make_model():
    model = tf.keras.Sequential()

    # model.add(layers.Resizing(32, 32, interpolation="bilinear"))
    model.add(layers.Conv2D(32, (3, 3), input_shape=[38, 33, 1]))
    # assert model.output_shape == (None, 36, 31, 32)
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(64, (3, 3)))
    # assert model.output_shape == (None, 34, 29, 64)
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='tanh'))

    return model

model = make_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

@tf.function
def train_step(images, compare_to):
	with tf.GradientTape() as tape:
		output = model(images, training=True)

		loss = cross_entropy(compare_to(output), output)

	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return loss

BATCH_SIZE = 16

def train(epochs):
    for epoch in range(0, epochs):
        start = time.time()

        # disc_losses = []
        # gen_losses = []

        for batch in range(8192 // BATCH_SIZE):
            audio_batch = data.get_batch(BATCH_SIZE)
            print(audio_batch)
            loss = train_step(audio_batch, tf.ones_like)
            print(f"Batch {batch} of epoch {epoch}: loss: {loss}", end="\r")
        
        # avg_gen = np.mean(gen_losses)
        # avg_disc = np.mean(disc_losses)

        # print (f'Epoch {epoch} ({8096} images); Time: {time.time()-start} sec; avg gen loss {avg_gen}; avg disc loss {avg_disc}')
        
        # if (epoch + 1) % 8 == 0:
        #     print("Saving checkpoint...")
        #     checkpoint.save(file_prefix = checkpoint_prefix)
        
train(10)
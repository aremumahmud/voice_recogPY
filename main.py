import pyaudio
import wave
import numpy as np
import pydub

def load_audio_from_avi(avi_file_path):
    # Load the AVI file using pydub
    video = pydub.AudioSegment.from_file(avi_file_path)

    # Extract the audio data from the video
    audio_data = np.frombuffer(video.raw_data, dtype=np.int16)

    return audio_data


filePaths = ["v1.aac","v2.aac","v3.aac"]
audioData = [load_audio_from_avi(path) for path in filePaths]

sampleAudio = load_audio_from_avi("v1.aac")


import tensorflow as tf
import numpy as np
import random

def add_noise(audio_data):
    # Add random noise to the audio data
    noise_signal = np.random.randn(len(audio_data))
    noisy_audio_data = audio_data + noise_signal
    return noisy_audio_data

def pitch_shift(audio_data, pitch_shift_factor):
    # Apply pitch shifting to the audio data
    pitched_audio_data = []
    for sample in audio_data:
        pitched_sample = sample * pitch_shift_factor
        pitched_audio_data.append(pitched_sample)
    pitched_audio_data = np.array(pitched_audio_data)
    return pitched_audio_data

def speed_shift(audio_data, speed_shift_factor):
    # Apply speed shifting to the audio data
    speed_shifted_audio_data = []
    for sample in audio_data:
        speed_shifted_sample = sample / speed_shift_factor
        speed_shifted_audio_data.append(speed_shifted_sample)
    speed_shifted_audio_data = np.array(speed_shifted_audio_data)
    return speed_shifted_audio_data

def spectral_perturbations(audio_data):
    # Convert audio data to frequency domain using FFT
    fft_result = np.fft.fft(audio_data)

    # Apply random phase shift
    random_phase = np.random.uniform(0, 2 * np.pi, size=fft_result.shape)
    phase_shifted_fft = fft_result * np.exp(1j * random_phase)

    # Apply random gain modification
    random_gain = np.random.uniform(0.8, 1.2, size=fft_result.shape)
    gain_modified_fft = phase_shifted_fft * random_gain

    # Apply time-stretching
    time_stretch_factor = np.random.uniform(0.8, 1.2)
    time_stretched_fft = np.fft.ifftshift(np.fft.fftshift(gain_modified_fft) / time_stretch_factor)

    # Convert back to time domain using IFFT
    spec_perturbed_audio_data = np.real(np.fft.ifft(time_stretched_fft))

    return spec_perturbed_audio_data

def verify_voice(training_audio_data, sample_voice_data):
    # Convert audio data to mel-spectrograms
    mel_spectrograms = []
    for audio in training_audio_data + [sample_voice_data]:
        # Apply data augmentation
        augmented_audio = add_noise(audio)
        pitched_audio = pitch_shift(audio, 1.2)
        speed_shifted_audio = speed_shift(audio, 0.8)
        spec_perturbed_audio = spectral_perturbations(audio)

        # Extract mel-spectrogram features
        mel_spec = tf.signal.mel_spectrogram(audio, 224, 14)

        # Append features from original and augmented samples
        mel_spectrograms.append(mel_spec)
        mel_spectrograms.append(tf.signal.mel_spectrogram(augmented_audio, 224, 14))
        mel_spectrograms.append(tf.signal.mel_spectrogram(pitched_audio, 224, 14))
        mel_spectrograms.append(tf.signal.mel_spectrogram(speed_shifted_audio, 224, 14))
        mel_spectrograms.append(tf.signal.mel_spectrogram(spec_perturbed_audio, 224, 14))

    # Train a classifier using the extracted mel-spectrograms
    classifier_model = tf.keras.Sequential([
        # 2D convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 14, 1)),

        # 2D max pooling layer with 2x2 pool size
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten the output of the max pooling layer
        tf.keras.layers.Flatten(),

        # Fully connected layer with 128 neurons and ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),

        # Dropout layer to prevent overfitting
        tf.keras.layers.Dropout(0.5),

        # Output layer with 2 neurons representing the two possible speaker identities (same or different)
        # Use sigmoid activation for binary classification
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])

    classifier_model.compile(optimizer='adam', loss='categoricalCrossentropy', metrics=['accuracy'])

    # Convert mel-spectrograms to NumPy arrays
    mel_spectrograms_np = np.array(mel_spectrograms)

    # Train the classifier
    classifier_model.fit(mel_spectrograms_np[:-1], [1] * (len(training_audio_data) + 4), epochs=10)

    # Predict the speaker identity for the sample voice
    prediction = classifier_model.predict(mel_spectrograms_np[-1])
    is_same_voice = prediction[0] > 0.5

    return is_same_voice


isVoice = verify_voice(audioData,sampleAudio)
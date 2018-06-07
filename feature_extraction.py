# coding= UTF-8

import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf

def feature_extraction(file_name):
    X, sample_rate = librosa.load(file_name)
    if X.ndim > 1:
        X = X[:, 0]
    X = X.T

    # Get features
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)  # 40 values
    # zcr = np.mean(librosa.feature.zero_crossing_rate)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, # tonal centroid features
                      axis=0)

    # Return computed features
    return mfccs, chroma, mel, contrast, tonnetz


# Process audio files: Return arrays with features and labels
def parse_audio_files(parent_dir, sub_dirs, file_ext='*.ogg'):  ## .ogg audio format
    features, labels = np.empty((0, 193)), np.empty(0)  # 193 features total. This can vary

    for label, sub_dir in enumerate(sub_dirs):  ##Enumerate() function adds a counter to an iterable.
        for file_name in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):  ##parent is audio-data, sub_dirs are audio classes
            try:
                mfccs, chroma, mel, contrast, tonnetz = feature_extraction(file_name)
            except Exception as e:
                print("[Error] there was an error in feature extraction. %s" % (e))
                continue

            extracted_features = np.hstack(
                [mfccs, chroma, mel, contrast, tonnetz])  # Stack arrays in sequence horizontally (column wise)
            features = np.vstack([features, extracted_features])  # Stack arrays in sequence vertically (row wise).
            labels = np.append(labels, label)
        print("Extracted features from %s, done" % (sub_dir))
    return np.array(features), np.array(labels, dtype=np.int)

# Read sub-directories (audio classes)
audio_directories = os.listdir("audio-data/")
audio_directories.sort()

# Function call to get labels and features
# This sabes a feat.npy and label.npy numpy-files in the current directory
features, labels = parse_audio_files('audio-data', audio_directories)
np.save('feat.npy', features)
np.save('label.npy', labels)
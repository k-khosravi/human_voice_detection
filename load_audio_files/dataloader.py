import librosa
import os
from config.config import data_path, data_vectors_path, max_len
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from tqdm import tqdm


class DataLoader():
    def __init__(self):
        self.data_dir = data_path
        self.data_vactors_dir = data_vectors_path
        self.max_len = max_len
        self.labels = self.get_labels()

    def load_wave_dataset(self):
        data = self.prepare_dataset()
        dataset = []
        for label in data:
            for mfcc in data[label]['mfcc']:
                dataset.append((label, mfcc))
        self.dataset = dataset[:100]
        # return dataset[:100]

    def get_labels(self):
        labels = os.listdir(self.data_dir)
        # label_indices = np.arange(0, len(labels))
        # return labels, label_indices, to_categorical(label_indices)
        return labels

    def prepare_dataset(self):
        data = {}
        for label in self.labels:
            data[label] = {}
            data[label]['path'] = [self.data_dir + label + '/' + wavfile for wavfile in os.listdir(self.data_dir + '/' + label)]
            vectors = []

            # load .wav file and convert to mfcc
            for wavfile in data[label]['path']:
                wave, sr = librosa.load(wavfile, mono=True, sr=None)
                # Downsampling
                wave = wave[::3]
                # mfcc = librosa.feature.mfcc(wave, sr=16000)
                mfcc = self.wav2mfcc(wave)
                vectors.append(mfcc)
            data[label]['mfcc'] = vectors
        return data

    def wav2mfcc(self, wave):
        mfcc = librosa.feature.mfcc(wave, sr=16000)
        # If maximum length exceeds mfcc lengths then pad the remaining ones
        if (max_len > mfcc.shape[1]):
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        # Else cutoff the remaining parts
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc

    def save_data_to_array(self):
        vectors = {}
        for label in self.labels: vectors[label] = []
        for wave_data in self.dataset:
            vectors.get(wave_data[0]).append(wave_data[1])
        for label in self.labels:
            np.save(data_vectors_path+label + '.npy', vectors.get(label))

    def get_train_test(self, split_ratio=0.6, random_state=42):
        # Getting first arrays
        X = np.load(self.labels[0] + '.npy')
        y = np.zeros(X.shape[0])

        # Append all of the dataset into one single array, same goes for y
        for i, label in enumerate(labels[1:]):
            x = np.load(label + '.npy')
            X = np.vstack((X, x))
            y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

        assert X.shape[0] == len(y)
        return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)


# if you run this file all audio files in /data/ dir load-process-convert to mfcc-save as .npy files
if __name__ == '__main__':
    data_loader = DataLoader()
    data_loader.load_wave_dataset()
    data_loader.save_data_to_array()

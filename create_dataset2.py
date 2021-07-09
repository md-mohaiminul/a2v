import os
import glob
import pandas as pd
from moviepy.editor import *
import skvideo.io
import numpy as np
import cv2
import librosa
import pickle

fps = 16
n_fft = 1024  # window size
hop_length = 512
sr = 16000
length = 2

dir = '/playpen/mohaiminul/Snoring_selected'
files = glob.glob(os.path.join(dir, '*.mp4'))

video = skvideo.io.vread(files[0])

k = 0
for file in files:
    clip = VideoFileClip(file)
    audio, sample_rate = librosa.load(file, sr=16000)
    print(clip.duration)
    if(clip.duration < 2):
        continue
    for start in range(0, int(clip.duration),length):
        dict = {}
        images = []
        for i in range(start*fps, (start+length)*fps):
            img = clip.get_frame(i / clip.duration)
            img = cv2.video_frame = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
        images = np.asarray(images)
        images = np.expand_dims(images, axis=0) / 255.0
        dict["video"] = images

        audio_clip = audio[start * sr:(start + length) * sr]
        stft = librosa.core.stft(audio_clip, hop_length=hop_length, n_fft=n_fft)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        normalized = librosa.util.normalize(log_spectrogram)
        normalized = np.transpose(normalized, (1, 0))
        if normalized.shape != (63, 513):
            continue
        dict["audio"] = normalized
        a_file = open('./dataset/dict_' + str(k)+ '.pkl', 'wb')
        pickle.dump(dict, a_file)
        a_file.close()
        print(k, dict["audio"].shape, dict["video"].shape)
        k += 1
import os
import glob
import pandas as pd
from moviepy.editor import *
import skvideo.io
import numpy as np
import cv2
import librosa
import imageio

duration =1
video_length = 16
audio_length = 16000
n_fft = 1024  # window size
hop_length = 512
sr = 16000

dir = '/playpen/mohaiminul/Snoring_selected'
files = glob.glob(os.path.join(dir, '*.mp4'))

file_count = 0
for file in files:
    video = skvideo.io.vread(file)
    for i in range(int(video.shape[0]/ video_length)):
        frames = []
        for frame in range(i * video_length, (i + 1) * video_length):
            image = cv2.resize(video[frame], (64, 64), interpolation=cv2.INTER_AREA)
            grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #for gray image
            frames.append(grayimage)
        frames = np.asarray(frames)
        videoarray = np.expand_dims(frames, axis=0)  #for gray image
        #video_frames = video_frames.transpose(3, 0, 1, 2) / 255.0 #for color image
        print(i, file_count, videoarray.shape)
        np.save('./dataset/video_' + str(file_count) + '.npy', videoarray)
        file_count += 1
    if file_count>=1000:
        break

    # audio, sample_rate = librosa.load(file, sr=16000)
    # if (len(audio) > audio_length):
    #     start = int(len(audio) / 2 - audio_length / 2)
    #     audio = audio[start:start + audio_length]
    # else:
    #     x = np.zeros(audio_length - len(audio))
    #     audio = np.concatenate((audio, x), axis=0)
    # stft = librosa.core.stft(audio, hop_length=hop_length, n_fft=n_fft)
    # spectrogram = np.abs(stft)
    # log_spectrogram = librosa.amplitude_to_db(spectrogram)
    # np.save('./dataset/audio_' + str(k) + '.npy', log_spectrogram)
    # k += 1

# ALternative
# video = VideoFileClip(file)
#     if video.duration <duration:
#         start = 0
#         end = video.duration
#     else:
#         start = video.duration / 2 - duration / 2
#         end = video.duration / 2 + duration / 2
#
#     video_frames = []
#     for i in range(video_length):
#         video_frame = video.get_frame(start + i * (end - start) / video_length)
#         video_frame = cv2.resize(video_frame, (96, 96), interpolation=cv2.INTER_AREA)
#         video_frames.append(video_frame)
#     video_frames = np.asarray(video_frames)
#     video_frames = video_frames.transpose(3, 0, 1, 2) / 255.0
#     np.save('./dataset/video_'+str(k)+'.npy', video_frames)
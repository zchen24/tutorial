#!/usr/bin/env python3

"""
A script that loads a wav file and plot

Author: Zihan Chen
Date: 2021-11-25
"""


import wave
import struct
import matplotlib.pyplot as plt


audio_clips = ['./audio-1.wav',
               './audio-2.wav']


all_audio_data = []
for a in range(len(audio_clips)):
    clip = audio_clips[a]
    wave_file = wave.open(clip, 'r')

    channels = wave_file.getnchannels()
    frame_rate = wave_file.getframerate()
    num_frames = wave_file.getnframes()

    data = []
    for i in range(num_frames-1):
        frame = wave_file.readframes(1)
        data.append(struct.unpack('h', frame[0:2])[0])

    all_audio_data.append(data)


n0 = 0
n1 = 200000
for i in range(len(audio_clips)):    
    plt.subplot(len(audio_clips),1,i)
    plt.plot(all_audio_data[0][n0:n1])
    plt.grid(True)
    plt.ylim([-25000, 25000])
    plt.ylabel('Audio')

plt.show()

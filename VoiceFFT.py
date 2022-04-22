import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import keyboard
import librosa
#import pyrubberband
CHANNELS = 1
RATE = 44100
SAMPLE = 1024*8
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                output=False,
                input=True)
streamOut = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=False)

stream.start_stream()

fig, ax = plt.subplots()
note = 0

from tkinter import *

master = Tk()
w1 = Scale(master, from_=-30, to=30)
w1.pack()
w2 = Scale(master, from_=1024, to=1024*8, orient=HORIZONTAL)
w2.pack()

while stream.is_active():
    master.update()
    print(w1.get())
    data = stream.read(w2.get())
    #print(len(data))
    data = np.frombuffer(data,dtype=np.float32)
    data*3
    data = librosa.effects.pitch_shift(data,sr=44100, n_steps=int(w1.get()), bins_per_octave=24)
    #data = pyrubberband.pyrb.pitch_shift(data,44100, n_steps=note)

    #print(len([x for x in data if np.isnan(x)]))
    #print(len(data))
    #print(data[300:400])

    if keyboard.is_pressed('1'):
        note +=1
    if keyboard.is_pressed('2'):
        note -=1
    print(note)
    #
    # ik = np.array([2j*np.pi*(note)*k for k in range(1024)]) / 1024
    # shi = np.exp(ik)
    #
    # data = data*shi
    #
    # data = np.fft.fft(data)
    # # for i in range(1024):
    # #     print(i)
    #     # for j in range(1024):
    #     #     pass
    # temp = data[512]
    # data[512] = 20+20j
    # ax.plot(np.fft.fftfreq(SAMPLE,1/44100),np.abs(data.real))
    # plt.pause(0.001)
    # plt.cla()
    # data[512] = temp
    #
    # data = np.fft.ifft(data)
    # data = np.array(data, dtype='short')
    # data = data.tostring()


    streamOut.write(data, w2.get())

stream.stop_stream()
print("Stream is stopped")
stream.close()
p.terminate()

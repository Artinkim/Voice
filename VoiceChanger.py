import pyaudio
import numpy as np
import time
from matplotlib import pyplot as plt
import matplotlib
p = pyaudio.PyAudio()
import librosa
import keyboard
import scipy.fftpack as sfft
CHANNELS = 1
RATE = 44100
SAMPLE = 1024
def callback(in_data, frame_count, time_info, flag):
    # using Numpy to convert to array for processing
    # audio_data = np.fromstring(in_data, dtype=np.float32)
    return in_data, pyaudio.paContinue

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
                # stream_callback=callback)
def shift(arr, num, fill_value):
    arr = np.roll(arr,num)
    if num < 0:
        arr[num:] = fill_value
    elif num > 0:
        arr[:num] = fill_value
    return arr
def skip(arr,num,fill_value):
    for i,val in enumerate(arr):
        if i%num!= 0:
            arr[i]=0
    return arr
def weightedSkip(arr,num):
    r = np.arange(0,(np.pi/num)*SAMPLE,np.pi/num)
    #print(r[:10])
    m = (np.cos(r)+1)/2
    #print(m[:10])
    return arr*m

f = []
curr = 16.35
for i in range(120):
    f.append(curr)
    curr*=2**(1/12)

def autotuneFFT(c):
    l = []
    for i in range(120):
        l.append(c)
        c*=2 #**(1/12)
    freq = np.fft.fftfreq(SAMPLE)*44100
    #print(l)
    a=[]
    j=0
    b = 0
    t = l[j]
    for i in range(SAMPLE):
        while freq[i] > t and j<len(l)-1:
            b=t
            j+=1
            t=l[j]
        #print(b,freq[i],t,i,j)
        x = abs(b-freq[i])
        y = abs(t-freq[i])
        a.append(abs((x-y)/(x+y)))
        #print(freq[15:512])
        # for j in range():
        #     diff = abs(l[j]-freq[i])
        #     if diff < min1:
        #         min2 = min1
        #         min1 = diff
        #     elif diff < min2:
        #         min2 = diff
        # a.append(abs(min1-min2))
    #print(a)
    return a
stream.start_stream()
#time.sleep(1)
#print(autotuneFFT())
#im = plt.imshow(data, cmap='gist_gray_r', vmin=0, vmax=1)
#fig = plt.figure()
z =0
def ft(arr,bins):
    a = np.empty((len(bins)),np.float32)
    for freq in bins:
        for i,point in enumerate(arr):
            angle = -2j*np.pi*freq*i/44100
            a[0]+= point ** angle
        #print(freq)
    return a
def ift(arr,bins):
    a = np.empty((SAMPLE),np.float32)
    for freq in bins:
        w = np.arange(0,SAMPLE*freq*(2*np.pi)/44100,freq*(2*np.pi)/44100)
        #print(len(w))
        b = np.sin(w)
        #print(len(b),len(a),len(arr))
        a+=b
    return a
def m(data,bins):
    for i in data:
        bins.searchsorted(i)

#np.searchsorted(arr,freq)
#print(np.fft.fftfreq(SAMPLE)[:512]*44100)
note = 1 #16.35
fig, ax = plt.subplots()
while stream.is_active():
    data = stream.read(SAMPLE)
    # print(data[:100],"before")
    data = np.frombuffer(data,dtype=np.float32)

    #data = np.sin(np.arange(0,SAMPLE*0.1*np.pi,np.pi*0.1))+np.sin(np.arange(0,SAMPLE*0.9*np.pi,np.pi*0.9))
    #print(data)
    # z+=1
    # if z > 100:
    #     fig, ax = plt.subplots()
    #     ax.plot(np.fft.fftfreq(SAMPLE,1/44100)[:512], 2.0/SAMPLE * np.abs(data[:SAMPLE//2])) #np.fft.fftfreq(SAMPLE,1/44100)[:512]
    #     #plt.plot(data)
    #     plt.show()

    if keyboard.is_pressed('1'):
        note +=1 #note*(2**(1/12))
    if keyboard.is_pressed('2'):
        note -=1 #note*(2**(-1/12))
    print(note)

    #data = np.multiply(data,)
    ik = np.array([2j*np.pi*(note/34)*k for k in range(1024)]) / 1024
    shi = np.exp(ik)

    data = data*shi
    #data = data/shi

    data = np.fft.fft(data)
    # for i in range(len(data)):
    #     data[i] = data[i//2]
    #print(f)
    #data = ft(data,f)
    #print(data)
    # for i in range(20):
    #     data[i] = 0

    #fig.clear()

    #print(data)
    #data = ift(data,f)

    #plt.pause(0.0001)
    #print(data)
    #data = shift(data,0,0)
    #print(autotuneFFT())
    #data*=autotuneFFT(note)
    #data = weightedSkip(data,2)
    #print(data)
    data*=3

    # for i in range(len(data)):
    #     data[i] = np.multiply(data[i], np.exp(1j*np.pi*(note/22050)*i))
    # print(data)

    temp = data[512]
    data[512] = 99+99j
    ax.plot(np.fft.fftfreq(SAMPLE,1/44100),np.abs(data.real)) #np.fft.fftfreq(SAMPLE,1/44100)[:512]
    plt.pause(0.001)
    plt.cla()
    data[512] = temp

    #plt.close()
    #plt.close()
    #plt.close()
    #plt.show()
    #time.sleep(1)
    #plt.close()
    # if z > 100:
    #     fig, ax = plt.subplots()
    #     ax.plot(np.fft.fftfreq(SAMPLE,1/44100)[:512],np.abs(data[:SAMPLE//2].real)) #np.fft.fftfreq(SAMPLE,1/44100)[:512]
    #     #plt.plot(data)
    #     plt.show()
    data = np.fft.ifft(data)
    #data = data/shi
    # if z > 100:
    #     fig, ax = plt.subplots()
    #     ax.plot(np.fft.fftfreq(SAMPLE,1/44100)[:512], 2.0/SAMPLE * np.abs(data[:SAMPLE//2]))
    #     plt.show()
    #     #time.sleep(1)
    #     z=0
    # for i in range(100):
    #     data[1023-i] = 0
    data = np.array(data, dtype='float32')
    #data = data/shi
    #for i in range():
    #data = data[::2]

    data = data.tostring()
    # data = np.recarray(data)
    # data.view(np.recarray([(SAMPLE)],dtype=np.float32))
    #print(data)

    streamOut.write(data, SAMPLE)


#ani = matplotlib.animation.FuncAnimation(fig, animate, interval=1)
#plt.show()
stream.stop_stream()
print("Stream is stopped")

stream.close()

p.terminate()

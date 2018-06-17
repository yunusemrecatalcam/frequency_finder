import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import struct
import time

nFFT = 512
BUF_SIZE = 4 * nFFT
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
              channels=CHANNELS,
              rate=RATE,
              input=True,
              frames_per_buffer=BUF_SIZE)

MAX_y = 2.0 ** (p.get_sample_size(FORMAT) * 8 - 1)

class Yvars:
    lastY = np.zeros(511,dtype=np.float64)
    diffY = np.zeros(511,dtype=np.float64)

fig =plt.gcf()
fig.show()
fig.canvas.draw()

arr=np.array([1,3,5,7,9])

while 1:

    N = max(stream.get_read_available() / nFFT, 1) * nFFT
    data = stream.read(N)

    y = np.array(struct.unpack("%dh" % (N * CHANNELS), data)) / MAX_y
    y_L = y[::2]
    y_R = y[1::2]

    Y_L = np.fft.fft(y_L, nFFT)
    Y_R = np.fft.fft(y_R, nFFT)

    # Sewing FFT of two channels together, DC part uses right channel's
    Y = np.abs(np.hstack((Y_L[-nFFT / 2:-1], Y_R[:nFFT / 2])))
    Yvars.diffY = (Y -Yvars.lastY)
    hp= np.sum(np.abs(Yvars.diffY[290:350]))
    print hp
    Yvars.lastY = Y
    if hp > 1.0:
        print "ov my whistle detected dafdmdmj"
    plt.plot(np.linspace(-20000,20000,num=len(Yvars.diffY)),Yvars.diffY)
    plt.ylim([-1,1])
    fig.canvas.draw()
    plt.clf()

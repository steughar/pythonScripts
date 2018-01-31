import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
matplotlib.use('TkAgg')

img = mpimg.imread('cameraman.png')


def rgb2grayscale(rgb):
    return np.dot(rgb[..., : 3], [0.299, 0.587, 0.114])


gray_image = rgb2grayscale(img)

img = np.array(([1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1], [2, 4, 4, 2]))
print(img)

print(img[0][3], "value")
print(img[0][1]-img[0][3], "difference")
plt.imshow(img, cmap='gray')
# plt.show()


def haarWavelet(vectorData):
    wavelet_summ = np.zeros(int(len(vectorData[0])*len(vectorData[0])/2))
    wavelet_diff = np.zeros(int(len(vectorData[0])*len(vectorData[0])/2))
    n = 0
    for i in range(0, len(vectorData)):
        print(vectorData[i])
        for k in range(0, len(vectorData[i]), 2):
                print(vectorData[i][k], vectorData[i][k+1])
                wavelet_summ[n] = vectorData[i][k] + vectorData[i][k+1]
                wavelet_diff[n] = vectorData[i][k] - vectorData[i][k+1]
                n += 1
 #   print(wavelet_summ/2, wavelet_diff/2)
    return wavelet_summ, wavelet_diff


# wawelet by rows at first
wavelet_summ, wavelet_diff = haarWavelet(img)
print(wavelet_summ, wavelet_diff)


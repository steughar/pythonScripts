import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import os
cwd = os.getcwd()
files = os.listdir(cwd)
print("Files in %s : %s" % (cwd, files))
matplotlib.use('TkAgg')

img = mpimg.imread('cameraman.png')


def rgb2grayscale(rgb):
    return np.dot(rgb[..., : 3], [0.299, 0.587, 0.114])


gray_image = rgb2grayscale(img)

#  [x0, x1, x2, x3, x4, x5] =>
#  => [x0 + x1, x2 + x3, x4 + x5]|[x0 - x1, x2 - x3, x4- x5] =>
#  [s0, s1, s2]|[d0, d1, d2] : s - summ ; d - difference
def haarWaveletHorizontal(vectorData):
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


#  [x0, x1, x2, x3, x4, x5]^(T) =>
#  => [x0 + x1, x2 + x3, x4 + x5]|[x0 - x1, x2 - x3, x4- x5] =>
#  [s0, s1, s2]|[d0, d1, d2] : s - summ ; d - difference
def haarWaveletVertical(vectorData):
    wavelet_summ = np.zeros(int(len(vectorData[0])*len(vectorData[0])/2))
    wavelet_diff = np.zeros(int(len(vectorData[0])*len(vectorData[0])/2))
    n = 0
    for i in range(0, len(vectorData)):
        print(vectorData[i])
        for k in range(0, len(vectorData[i]), 2):
            print(vectorData[i][k], vectorData[i][k+1])
            wavelet_summ[n] = vectorData[k][i] + vectorData[k+1][i]
            wavelet_diff[n] = vectorData[k][i] - vectorData[k+1][i]
            n += 1
#   print(wavelet_summ/2, wavelet_diff/2)
    return wavelet_summ, wavelet_diff


# wawelet by rows at first
wavelet_summ, wavelet_diff = haarWaveletHorizontal(gray_image)


new_img = gray_image

#  concatenate array of diff and summ and stacking them one by one from top to
#  bottom
shift = 0
for i in range(0, 256):
    new_img[i][0:128] = wavelet_summ[shift:shift+128]
    new_img[i][128:256] = wavelet_diff[shift:shift+128]
    shift += 128


plt.imshow(new_img, cmap='gray')
plt.show()

plt.imshow(rgb2grayscale(img), cmap='gray')
plt.show()


wavelet_summ, wavelet_diff = haarWaveletVertical(gray_image)

shift = 0
for i in range(0, 256):
    new_img[i][0:128] = wavelet_summ[shift:shift+128]
    new_img[i][128:256] = wavelet_diff[shift:shift+128]
    shift += 128

new_img = np.transpose(new_img)

plt.imshow(new_img, cmap='gray')
plt.show()

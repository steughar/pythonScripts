import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
matplotlib.use('TkAgg')

img = mpimg.imread('cameraman.png')
print("image")
print(img[:, 0])


def rgb2grayscale(rgb):
    return np.dot(rgb[..., : 3], [0.299, 0.587, 0.114])


gray = rgb2grayscale(img)


def haar1D(n, SIZE):
    # check power of two
    if math.floor(math.log(SIZE) / math.log(2)) != math.log(SIZE) / math.log(2):
        print("Haar defined only for lengths that are a power of two")
        return None
    if n >= SIZE or n < 0:
        print("invalid Haar index")
        return None

    # zero basis vector
    if n == 0:
        return np.ones(SIZE)

    # express n > 1 as 2^p + q with p as large as possible;
    # then k = SIZE/2^p is the length of the support
    # and s = qk is the shift
    p = math.floor(math.log(n) / math.log(2))
    pp = int(pow(2, p))
    k = SIZE / pp
    s = (n - pp) * k

    h = np.zeros(SIZE)
    h[int(s):int(s+k/2)] = 1
    h[int(s+k/2):int(s+k)] = -1
    # these are not normalized
    return h


def haar2D(n, SIZE=8):
    # get horizontal and vertical indices
    hr = haar1D(n % SIZE, SIZE)
    hv = haar1D(int(n / SIZE), SIZE)
    # 2D Haar basis matrix is separable, so we can
    #  just take the column-row product
    H = np.outer(hr, hv)
    H = H / math.sqrt(np.sum(H * H))
    return H


tx_img = np.zeros(256*256)
for k in range(0, (256*256)):
    tx_img[k] = np.sum(gray*haar2D(k, 256))
print("analysis routine is done")

lossy_gray = np.copy(tx_img)
lossy_gray[int(len(tx_img)/16):] = 0

gray_rx = np.zeros((256, 256))
for k in range(0, (256*256)):
    gray_rx += lossy_gray[k]*haar2D(k, 256)
print("synthesis routine is done")

plt.imshow(img, cmap='gray')
plt.show()
plt.imshow(gray_rx, cmap='gray')
plt.show()

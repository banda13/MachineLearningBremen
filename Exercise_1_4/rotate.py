import cv2
from scipy import ndimage, misc

import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('city.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum_original = np.log(np.abs(fshift))


img_rotated = ndimage.rotate(img, 45, reshape=False)
f_rot = np.fft.fft2(img_rotated)
fshift_rot = np.fft.fftshift(f_rot)
magnitude_spectrum_rotated = np.log(np.abs(fshift_rot))




plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum_original, cmap = 'gray')
plt.title('Originals Magnitude'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img_rotated, cmap = 'gray')
plt.title('Rotated Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(magnitude_spectrum_rotated, cmap = 'gray')
plt.title('Rotated Magnitude'), plt.xticks([]), plt.yticks([])

plt.show()
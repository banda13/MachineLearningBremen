import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('asguard2.png',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift))
f2=f
f3=f
frequency = 380
frequency2 = 2
a = int(np.shape(magnitude_spectrum)[0]/2)
b = int(np.shape(magnitude_spectrum)[1]/2)

for i in range(np.shape(f2)[0]):
    for j in range(np.shape(f2)[1]):
        if (pow(i-a,2)+pow(j-b,2)) < pow(frequency,2):
        #if (pow(i-a,2)/pow(a,2)+pow(j-b,2)/pow(b,2))<frequency or (pow(i-a,2)/pow(a,2)+pow(j-b,2)/pow(b,2))>frequency2:
            f2[i,j] = 0

anti = np.real(np.fft.ifft2(f2))

for i in range(np.shape(magnitude_spectrum)[0]):
    for j in range(np.shape(magnitude_spectrum)[1]):
       if (pow(i-a,2)+pow(j-b,2)) > pow(frequency,2):
        #if (pow(i-a,2)/pow(a,2)+pow(j-b,2)/pow(b,2))<frequency or (pow(i-a,2)/pow(a,2)+pow(j-b,2)/pow(b,2))>frequency2:
            magnitude_spectrum[i,j] = 0


for i in range(np.shape(magnitude_spectrum)[0]):
    for j in range(np.shape(magnitude_spectrum)[1]):
       if (pow(i-a,2)+pow(j-b,2)) > pow(frequency,2):
        #if (pow(i-a,2)/pow(a,2)+pow(j-b,2)/pow(b,2))<frequency or (pow(i-a,2)/pow(a,2)+pow(j-b,2)/pow(b,2))>frequency2:
            magnitude_spectrum[i,j] = 0


plt.subplot(311),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(313),plt.imshow(anti, cmap = 'gray')
plt.title('After clean'), plt.xticks([]), plt.yticks([])
plt.subplot(312),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])


plt.show()  
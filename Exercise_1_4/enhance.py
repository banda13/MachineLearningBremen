import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__enhance__":
    # Original image is read
    img = cv2.imread('asguard2.png',0)
    # Fast Fourier Transform of the original image
    f = np.fft.fft2(img)
    # The Transform is shifted in order to center it
    # (If the image has a dimension of MxN, now the origin of the transform is
    # in the point of coordinates (x,y) = (M/2, N/2) of the spectrum)
    fshift = np.fft.fftshift(f)
    # Magnitude spectrum is obtained
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum_filtered = magnitude_spectrum.copy()
    r = 200

    # Central element of the trasform is obtained
    m = int(np.shape(fshift)[0])
    n = int(np.shape(fshift)[1])

    # Lowpass filter is applied
    for i in range(m):
        for j in range(n):
            # Equation of points outside the circle with radius r and centre in (m/2,n/2)
            if (pow(i-(m/2),2)+pow(j-(n/2),2)) > pow(r,2):
                # All the frequencies outside this circle are completely attenuated
                fshift[i,j] = 0
                # We do the same thing for the magnitude so we can display a new spectrum
                magnitude_spectrum_filtered[i,j] = 0

    # It's applied the opposite shift that we have done before and then the final image is obtained
    # doing the antitrasform of Fourier
    img_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))

    # Display the results
    plt.subplot(221),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(img, cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(magnitude_spectrum_filtered, cmap = 'gray')
    plt.title('Magnitude spectrum filtered'), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(img_filtered, cmap = 'gray')
    plt.title('Image filtered'), plt.xticks([]), plt.yticks([])
    plt.show()
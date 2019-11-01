import numpy as np
import numpy.fft as fft
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 


#read the image
img=mpimg.imread('asguard2.png')
#plot the image
imgplot = plt.imshow(img)
#print
plt.show()

imgTransform = fft.fft2(img)

imgplot = plt.imshow(imgTransform)
plt.show()

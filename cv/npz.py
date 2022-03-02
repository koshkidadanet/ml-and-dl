import numpy as np
import cv2

a = cv2.imread('hui.jpg',0)
print(a)
np.savez('cifra', a)

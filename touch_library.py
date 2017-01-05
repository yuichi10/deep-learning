import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# simple graph
"""
x = np.arange(0, 6, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()
"""

# show easy to see graph
"""
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin')
plt.plot(x, y2, linestyle = '--', label='cos') # draw by dashed line
plt.xlabel('x') # x axis label
plt.ylabel('y') # y axis label
plt.title('sin & cos') # title
plt.legend()
plt.show()
"""

# show image by pyplot
img = imread('image/Lenna.png')
plt.imshow(img)
plt.show()


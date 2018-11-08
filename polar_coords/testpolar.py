from topolar import topolar
from skimage.data import chelsea
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#img = np.array(Image.open('test.png'))[...,0]/255

img = chelsea()[...,0] / 255.

#img = np.array([100,150,200,255]).reshape((2,2))/255
#img = np.arange(9).reshape((3,3))/255


pol, (rads,angs) = topolar(img)

print(img.shape, angs.shape)
fig,ax = plt.subplots(2,1,figsize=(6,8))

ax[0].imshow(img, cmap=plt.cm.gray, interpolation='bicubic')

ax[1].imshow(pol, cmap=plt.cm.gray, interpolation='bicubic')

ax[1].set_ylabel("Radius in pixels")
ax[1].set_yticks(range(0, img.shape[0]+1,50))
ax[1].set_yticklabels(rads[::50].round().astype(int))

ax[1].set_xlabel("Angle in degrees")
#ax[1].set_xticks(range(0, img.shape[1]+1, 112))

ax[1].set_xticks(np.linspace(0,img.shape[1],9))
ax[1].set_xticklabels(range(0,361,45))
a=range(0, img.shape[1]+1, 50)
b=(angs[::50]*180/3.14159).round().astype(int)
print(len(a),type(b))

plt.show()
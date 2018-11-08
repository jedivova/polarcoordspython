import numpy as np
from scipy.ndimage.interpolation import geometric_transform

def topolar(img, order=3):
    """
    Transform img to its polar coordinate representation.

    order: int, default 1
        Specify the spline interpolation order.
        High orders may be slow for large images.
    """
    # max_radius is the length of the diagonal
    # from a corner to the mid-point of img.

    max_radius = 0.5*np.linalg.norm( img.shape )
    #print(img.shape)
    #print(max_radius)

    def transform(coords):

        #print("coords ",coords)
        # Put coord[1] in the interval, [-pi, pi]
        theta = 2*np.pi*coords[1] / (img.shape[1] - 1.)
        #print("theta ", theta)

        # Then map it to the interval [0, max_radius].
        #radius = float(img.shape[0]-coords[0]) / img.shape[0] * max_radius
        radius = max_radius * coords[0] / img.shape[0]
        #print("radius ", radius)
        i = 0.5*img.shape[0] - radius*np.sin(theta)
        j = radius*np.cos(theta) + 0.5*img.shape[1]
        #print(i,j)
        #print()

        if coords==(0,0):
            print(coords)
            print(theta, radius)
        return i,j

    polar = geometric_transform(img, transform, order=order)

    print(img.shape, polar.shape)
    rads = max_radius * np.linspace(0,1,img.shape[0])
    angs = np.linspace(0, 2*np.pi, img.shape[1])

    return polar, (rads, angs)
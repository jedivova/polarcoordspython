#https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system


import numpy as np
import scipy as sp
import scipy.ndimage
from skimage.data import chelsea
from PIL import Image

import matplotlib.pyplot as plt

def main():

    im = Image.open('square.png')

    im = im.convert('L')
    data = np.array(im)
    print(data[1,1])

    #data = chelsea()[...,0]

    plot_polar_image(data, origin=None)
    #plot_polar_image(data, (100,100))
    #plot_directional_intensity(data, origin=None)

    plt.show()


def plot_polar_image(data, origin=None):
    """Plots an image reprojected into polar coordinages with the origin
    at "origin" (a tuple of (x0, y0), defaults to the center of the image)"""
    polar_grid, r, theta = reproject_image_into_polar(data, origin)

    #polar_grid=polar_rotate(polar_grid,90)

    fig, ax = plt.subplots(2, 1, figsize=(6, 8))

    ax[0].imshow(polar_grid, cmap=plt.cm.gray)

    ax[1].imshow(data, cmap=plt.cm.gray)

    #fig.savefig('foo.png')
    plt.show()


def Get_polar_coords(IMG):
    ny, nx = IMG.shape

    print('IMG', IMG.shape)
    origin_x, origin_y = nx // 2, ny // 2
    A = np.arange(nx)
    B = np.arange(ny)

    if nx % 2 != 0:
        A -= origin_x

    else:
        A[0: origin_x] -= origin_x
        A[origin_x:] -= origin_x - 1

    if ny % 2 != 0:
        B -= origin_y
    else:
        B[0: origin_y] -= origin_y
        B[origin_y:] -= origin_y - 1
    Matrix = np.ones((origin_y, nx))*np.pi*2
    Matrix1= np.zeros((ny-origin_y, nx))
    c = np.vstack((Matrix, Matrix1))

    x_coords, y_coords = np.meshgrid(A, B)
    r = np.sqrt(x_coords ** 2 + y_coords ** 2)  # add /2 for easing calculations
    theta = np.arctan2(y_coords, x_coords)+c
    return r, theta


def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2

    # Determine that the min and max r and theta coords will be...

    r, theta = Get_polar_coords(data)


    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), ny)
    theta_i = np.linspace(theta.min(), theta.max(), nx)
    print(theta.min(), theta.max())
    print(r.min(), r.max())
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)

    # We need to shift the origin back to
    # back to the lower-left corner...
    if nx % 2 != 0:
        xi += origin_x
    else:
        xi[0: origin_x] += origin_x
        xi[origin_x:] += origin_x - 1

    if ny % 2 != 0:
        yi += origin_y
    else:
        yi[0: origin_y] += origin_y
        yi[origin_y:] += origin_y - 1



    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((yi, xi)) # (map_coordinates requires a 2xn array)

    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)

    zi = sp.ndimage.map_coordinates(data, coords, order=3)
    output = zi.reshape((ny, nx))

    print(output.shape)
    return output, r_i, theta_i


def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def polar_rotate(IMG, angle):
    width = IMG.shape[1]
    shift = int(round(angle/360*width))
    return np.roll(IMG, shift, axis=1)


if __name__ == '__main__':
    main()

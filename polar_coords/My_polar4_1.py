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

    data = chelsea()[...,0]

    plot_polar_image(data, origin=None)
    #plot_polar_image(data, (100,100))
    #plot_directional_intensity(data, origin=None)

    plt.show()


def plot_polar_image(data, origin=None):
    """Plots an image reprojected into polar coordinages with the origin
    at "origin" (a tuple of (x0, y0), defaults to the center of the image)"""
    polar_grid = reproject_image_into_polar(data, origin)

    #polar_grid=polar_rotate(polar_grid,45)

  #  fig, ax = plt.subplots(2, 1, figsize=(6, 8))

    #ax[0].imshow(polar_grid, cmap=plt.cm.gray)
   # ax[0].set_ylabel("Radius in pixels")
   # ax[0].set_xlabel("Angle in degrees")
   # ax[0].set_xticks(np.linspace(0, data.shape[1], 9))
   # ax[0].set_xticklabels(range(0, 361, 45))
  #  ax[1].imshow(data, cmap=plt.cm.gray)

    plt.figure()
    plt.imshow(polar_grid,cmap=plt.cm.gray)
    plt.axis('auto')
    #plt.ylim(plt.ylim()[::-1])
    plt.xlabel('Theta Coordinate (radians)')
    plt.ylabel('R Coordinate (pixels)')
    plt.title('Image in Polar Coordinates' )
    #fig.savefig('foo.png')
    plt.show()


def Get_polar_axes(shape,rad_width,phi_width):
    ny, nx = shape
    if ny % 2 != 0:
        theta_min = 0
    else:
        theta_min = np.arctan2(1, nx // 2)
    theta_max = np.arctan2(-1, nx // 2) + np.pi * 2
    #print("theta_mim-max", theta_min, theta_max)
    if ny % 2 == 0 and nx % 2 == 0:
        min_radius = np.sqrt(2)
    elif ny % 2 == 0 or nx % 2 == 0:
        min_radius = 1
    else:
        min_radius = 0
    max_radius = np.sqrt((nx // 2) ** 2 + (ny // 2) ** 2)
    #print("max_rad",min_radius, max_radius)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(min_radius, max_radius, rad_width)
    theta_i = np.linspace(theta_min, theta_max, phi_width)


    return r_i, theta_i


def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2

    r_i, theta_i = Get_polar_axes(data.shape, ny, nx)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)

    # We need to shift the origin back to
    # back to the lower-left corner...
    xi_mean = int(xi.shape[1]//2)+1
    yi_mean = int(yi.shape[0]//2)+1
    if nx % 2 != 0:
        xi += origin_x
    else:
        xi[0: xi_mean] += origin_x
        xi[xi_mean:] += origin_x - 1

    if ny % 2 != 0:
        yi += origin_y
    else:
        yi[0: yi_mean] += origin_y
        yi[yi_mean:] += origin_y - 1

    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((yi, xi)) # (map_coordinates requires a 2xn array)

    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)

    zi = sp.ndimage.map_coordinates(data, coords, order=3)
    output = zi.reshape((ny, nx))

    return output


def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def polar_rotate(IMG, angle):
    width = IMG.shape[1]
    shift = int(round(-angle/360*width))
    return np.roll(IMG, shift, axis=1)


if __name__ == '__main__':
    main()

import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import pandas as pd
import os
import multiprocessing
from My_polar5 import reproject_image_into_polar, polar_rotate

CPUnum=os.cpu_count()
term=10
number_of_images = 100
number_of_angles = int(360 / term)
#Create DataFrame from Image array


def Del_pixels(arr,percent):
    LIST=[]     #list_of_deleted_pixels
    shape = arr.shape
    N = int(shape[0]*shape[1]*percent)    #num of pixels to be deleted
    while len(LIST) != N:
        coord = np.random.randint(shape[0], size=2)  # choosing pixel
        if LIST.count((coord[0], coord[1])) == 0:
            LIST.append((coord[0], coord[1]))

    for i in range(len(LIST)):
        x,y=LIST[i]
        arr[x,y]=255
    #print("DEL DONE")
    return arr


def Generate(arr, Num):
    #angle = int(90*np.random.randint(3))           #Loses values on edjes of pic
    #rotated_arr = ndimage.rotate(arr, angle, reshape=False)
    angle = np.random.randint(4)
    rotated_arr = np.rot90(arr, angle)

    im = Image.fromarray(rotated_arr)
    im.save('2x2_{}.png'.format(Num))
    print("saved 2x2_{}.png".format(Num))
    return 1


def Generate1(arr, Num):
    #angle = int(90*np.random.randint(3))           #Loses values on edjes of pic
    #rotated_arr = ndimage.rotate(arr, angle, reshape=False)
    angle = np.random.randint(number_of_angles)*term
    rotated_arr = reproject_image_into_polar(arr, 100, 100)
    rotated_arr = polar_rotate(rotated_arr, angle)
    im = Image.fromarray(rotated_arr)
    if im.mode != 'L':
        im = im.convert('L')
    im.save('img{}.png'.format(Num))
    print("saved img{}.png".format(Num))

    return 1


def Multiprocess(arr,IMG_NUM):
    global CPUnum
    with multiprocessing.Pool(processes=CPUnum) as pool:
        # E_step
        E_results = [pool.apply_async(Generate1, (arr, i)) for i in range(IMG_NUM)]
        #print(E_results)
        #print(type(E_results))
        LOG_P = [(el.get()) for el in E_results]
        print(len(LOG_P))




if __name__ == '__main__':
    IMG_array = np.array(Image.open('test.png').convert('L'))
    #print(IMG_array[0:4,0:4])
    #IMG_array = np.array(Image.open('test.png'))[...,0]
    # record the original shape


    #Generate(IMG_array, 0.5,0)
    Multiprocess(IMG_array, number_of_images)
    im = Image.fromarray(np.zeros((100,100)))
    if im.mode != 'L':
        im = im.convert('L')
    im.save('img{}.png'.format(10))
#im = Image.fromarray(IMG_array)
#im.save("4x4.png")






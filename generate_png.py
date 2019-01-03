import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
#import pandas as pd
import os
from sys import platform
import multiprocessing
from My_polar5 import reproject_image_into_polar, polar_rotate, polar_with_del

CPUnum=os.cpu_count()
term=10
sigma = 10
number_of_images = 1000
number_of_angles = int(360 / term)
del_percent = 90
if platform == 'win32':
    way_to_img='imgs\img{}.png'
elif platform== 'linux':
    way_to_img = 'imgs/img{}.png'
#Create DataFrame from Image array


def Del_pixels(arr,percent):
    LIST=[]     #list_of_deleted_pixels
    shape = arr.shape
    N = int(shape[0]*shape[1]*percent/100)    #num of pixels to be deleted
    while len(LIST) != N:
        coord = np.random.randint(shape[0], size=2)  # choosing pixel
        if LIST.count((coord[0], coord[1])) == 0:
            LIST.append((coord[0], coord[1]))

    for i in range(len(LIST)):
        x,y=LIST[i]
        arr[x,y]=255
    #print("DEL DONE")
    plt.imshow(arr, cmap=plt.cm.gray)
    plt.show()

    return arr


def Del_pixels1(arr, percent):
    shape = arr.shape
    number_of_deleted=0
    for y in range(shape[0]):
        for x in range(shape[1]):
            if np.random.uniform() < percent/100:

                arr[y][x] = 0

                number_of_deleted+=1

    print('number of deleted',number_of_deleted)
    #plt.imshow(arr, cmap=plt.cm.gray)
    #plt.show()
    #print(np.count_nonzero(arr))
    return arr



def Generate(arr, Num):
    Num = Num
    #angle = int(90*np.random.randint(3))           #Loses values on edjes of pic
    #rotated_arr = ndimage.rotate(arr, angle, reshape=False)

    arr = put_normal_distribution(arr)
    angle = np.random.randint(number_of_angles)*term

    #polar_arr = reproject_image_into_polar(arr, 100, 100)
    polar_arr = polar_with_del(arr, 100, 100, del_percent)

    rotated_arr = polar_rotate(polar_arr, angle)
    im = Image.fromarray(rotated_arr)
    if im.mode != 'L':
        im = im.convert('L')
    im.save(way_to_img.format(Num))
    print("saved img{}.png".format(Num))
    return 1


def Generate1(arr, Num):

    #angle = int(90*np.random.randint(3))           #Loses values on edjes of pic
    #rotated_arr = ndimage.rotate(arr, angle, reshape=False)

    arr = put_normal_distribution(arr)
    arr = Del_pixels1(arr, del_percent)

    angle = np.random.randint(number_of_angles)*term
    polar_arr = reproject_image_into_polar(arr, 100, 100)
    rotated_arr = polar_rotate(polar_arr, angle)
    im = Image.fromarray(rotated_arr)
    if im.mode != 'L':
        im = im.convert('L')
    im.save(way_to_img.format(Num))
    print("saved img{}.png".format(Num))

    return 1


def Multiprocess(arr,IMG_NUM):
    global CPUnum
    with multiprocessing.Pool(processes=CPUnum) as pool:
        # E_step
        E_results = [pool.apply_async(Generate, (arr, i)) for i in range(IMG_NUM)]
        #print(E_results)
        #print(type(E_results))
        LOG_P = [(el.get()) for el in E_results]
        print(len(LOG_P))


def put_normal_distribution(IMG_array):
    shape = IMG_array.shape
    for y in range(shape[0]):
        for x in range(shape[1]):
            ii = np.random.normal(0, sigma)
            if (ii + IMG_array[y,x] <= 0):
                IMG_array[y,x] = 0
            elif (ii + IMG_array[y,x] < 255):
                IMG_array[y,x] = ii + IMG_array[y,x]
            else:
                IMG_array[y,x] = 255


    return IMG_array

def Generate_same_img(shape, num_of_img):
    arr = np.ones(shape)*100
    for i in range(num_of_img):
        im = Image.fromarray(arr)
        if im.mode != 'L':
            im = im.convert('L')
        im.save('img{}.png'.format(i))
        print("saved img{}.png".format(i))

if __name__ == '__main__':
    IMG_array = np.array(Image.open('test.png').convert('L'))


    #IMG_array = np.array(Image.open('test.png'))[...,0]


    #Generate_same_img(IMG_array.shape, 100)

    #Generate(IMG_array, 0.5,0)

    Multiprocess(IMG_array, number_of_images)

    #arr = reproject_image_into_polar(IMG_array, 100, 100)
    #rotated_arr = polar_rotate(arr, 180)
    #print(arr[0:5, 0:5])
    #im = Image.fromarray(rotated_arr)

    #if im.mode != 'L':
    #    im = im.convert('L')
    #im.save('test_img.png'.format(10))
#im = Image.fromarray(IMG_array)
#im.save("4x4.png")






import multiprocessing
import ctypes
import numpy as np
from PIL import Image
from scipy import ndimage
from math import sqrt, exp, log, pi, e
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from decimal import Decimal
from My_polar5 import reproject_image_into_polar, polar_rotate

#arr = multiprocessing.Array(ctypes.c_int8, 2 * 1 * 3)
CPUnum=os.cpu_count()
term=10
number_of_images = 100
number_of_angles = int(360 / term)
sigma = 1
shape = np.array(Image.open('img0.png')).shape
W=np.ones((number_of_angles))/number_of_images


def WeightsG(stepen,weigths):
    X=[]
    Len=stepen.shape[0]
    SUM=0
    #print(stepen)
    stepen -= stepen.max()  #i.e. *(e^x/e^x), where x is max degree

    for i in range(Len):
        X.append(Decimal(stepen[i]).exp()*Decimal(weigths[i]))
        SUM+=X[i]
    for i in range(Len):
        X[i] /= SUM

    arr=np.array(X)
    #print(arr)
    return arr, SUM


def E_step(number_of_image, priblijenie, weigths):
    #print("process {}".format(number_of_image))
    #print('calculating log(L) for {} picture'.format(number_of_image))
    #REF = priblijenie
    IMG = np.array(Image.open('img{}.png'.format(number_of_image)))

    step = np.zeros(number_of_angles)
    for i in range(number_of_angles):
        angle = i * term
        #print('angle = {}'.format(angle))
        #REF = ndimage.rotate(priblijenie, angle, reshape=False)

        #REF = np.rot90(priblijenie, -i)#при +i вертит против часовой
        REF = polar_rotate(priblijenie, -angle)

        #log_L = 0
        #for x in range(REF.shape[0]):
        #    for y in range(REF.shape[1]):
        #        log_L -= (int(IMG[x, y]) - int(REF[x, y])) ** 2/(2*sigma**2)

        a = (REF.astype(int) - IMG.astype(int)) ** 2/(2*sigma**2)
        log_L = -a.sum()
        step[i] = log_L
    vector, SUM = WeightsG(step,weigths)
    return vector, SUM



def M_step(number_of_image,vector,weights):
    #print('calculating pixels for {} picture {}'.format(number_of_image,vector))
    #IMG = np.array(Image.open('rot{}.png'.format(number_of_image)).convert('RGB'))

    IMG_arr = np.array(Image.open('img{}.png'.format(number_of_image)))
    priblij = np.zeros(IMG_arr.shape)

    for i in range(number_of_angles):
        angle = i * term
        #print('angle = {}'.format(angle))
        #REF = ndimage.rotate(IMG, -angle, reshape=False)
        IMG = polar_rotate(IMG_arr, angle)
        a = np.float64(vector[i]/weights[i])
        priblij += a * IMG
        #print(f"{vector[i]} * {REF[55,55]} = {priblij[55,55]}")
    #plt.imshow(priblij)
    #plt.show()
    return priblij



if __name__ == '__main__':
    priblij = np.array(Image.open('img0.png'))
    #priblij = np.array([6, 0, 30, 4]).reshape((2, 2))
    print(priblij[0:5, 0:5])
    print(priblij.sum())
    #print('priblij')
    #print(priblij)
    #print(type(priblij[0,0]))
    Weigths = np.ones((number_of_angles)) / number_of_angles

    SUM_LOG1=0
    for num_of_iter in range(20):
        print("iter {}".format(num_of_iter))
        with multiprocessing.Pool(processes=CPUnum) as pool:
            #E_step
            E_results = [pool.apply_async(E_step, (i, priblij, Weigths)) for i in range(number_of_images)]
            LOG_P = [(el.get()) for el in E_results]
            #print(LOG_P)

            VECT=[]
            SUM_LOG=0
            for i in range(number_of_images):
                VECT.append(LOG_P[i][0])
                SUM_LOG*=LOG_P[i][1]
            print(SUM_LOG)


 ###################       M_step           #
            #weight

            vectors = np.array(VECT)
            print(vectors[0].min())
            #print(vectors)
            Weigths=vectors.sum(0)/number_of_images
            #print(Weigths)


            priblij = np.zeros(shape)
            M_results = [pool.apply_async(M_step, (i, vectors[i], Weigths)) for i in range(number_of_images)]
            for el in M_results:
                priblij+=el.get()

            #print(priblij[50,50], priblij[60,60])
            #priblij = (priblij / number_of_images)
            priblij = (priblij / number_of_images/number_of_angles)
            priblij = priblij.astype('int32')     #normalize IMG

            print("current prediction is")
            print(priblij[0:5,0:5])
            print(priblij.sum())

            im = Image.fromarray(priblij)
            if im.mode != 'L':
                im = im.convert('L')
            im.save('Priblij{}.png'.format(num_of_iter))      #save img in png

            #if SUM_LOG1==SUM_LOG:
            #    break
            SUM_LOG1=SUM_LOG
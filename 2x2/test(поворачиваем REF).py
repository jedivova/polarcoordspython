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

#arr = multiprocessing.Array(ctypes.c_int8, 2 * 1 * 3)
CPUnum=os.cpu_count()
term=90
number_of_images = 10
number_of_angles = int(360 / term)
sigma = 1
shape = np.array(Image.open('2x2.png')).shape
W=np.ones((number_of_angles))/number_of_images


def WeightsG(stepen,weigths):
    X=[]
    Len=stepen.shape[0]
    SUM=0
    for i in range(Len):
        X.append(Decimal(stepen[i]).exp()*Decimal(weigths[i]))
        SUM+=X[i]
    for i in range(Len):
        X[i] /= SUM
    arr=np.zeros(Len)
    for i in range(Len):
        arr[i]=X[i]
    return arr, SUM.ln()


def E_step(number_of_image, priblijenie, weigths):
    #print("process {}")
    #print('calculating log(L) for {} picture'.format(number_of_image))
    #REF = priblijenie
    IMG = np.array(Image.open('2x2_{}.png'.format(number_of_image)))

    step = np.zeros(number_of_angles)
    for i in range(number_of_angles):
        #angle = i * term
        #print('angle = {}'.format(angle))
        #REF = ndimage.rotate(priblijenie, angle, reshape=False)

        REF = np.rot90(priblijenie, -i)#при +i вертит против часовой
        log_L = 0
        for x in range(REF.shape[0]):
            for y in range(REF.shape[1]):
                log_L = log_L - (int(IMG[x, y]) - int(REF[x, y])) ** 2/(2*sigma**2)
        step[i] = log_L

    vector, SUM = WeightsG(step,weigths)
    return vector, SUM


def M_step(number_of_image,vector,weights):
    #print('calculating pixels for {} picture {}'.format(number_of_image,vector))
    #IMG = np.array(Image.open('rot{}.png'.format(number_of_image)).convert('RGB'))

    IMG_arr = np.array(Image.open('2x2_{}.png'.format(number_of_image)))
    priblij = np.zeros(shape)
    for i in range(number_of_angles):
        angle = i * term
        #print('angle = {}'.format(angle))
        #REF = ndimage.rotate(IMG, -angle, reshape=False)

        IMG = np.rot90(IMG_arr, i)

        priblij += vector[i] * IMG/weights[i]
    #plt.imshow(priblij)
    #plt.show()
    return priblij


if __name__ == '__main__':
    #priblij = np.array(Image.open('2x2_0.png'))
    priblij = np.array([6, 0, 30, 4]).reshape((2, 2))
    print('priblij')
    print(priblij)
    print(type(priblij[0,0]))
    Weigths = np.ones((number_of_angles)) / number_of_images

    Log_L_old=0
    for num_of_iter in range(10):
        print("iter {}".format(num_of_iter))
        with multiprocessing.Pool(processes=CPUnum) as pool:
            #E_step
            E_results = [pool.apply_async(E_step, (i, priblij, Weigths)) for i in range(number_of_images)]
            LOG_P = [(el.get()) for el in E_results]
            print(LOG_P)

            #calculating Log Likelihood and copying weights vectors (g_ij)
            VECT=[]
            Log_L=0
            for i in range(number_of_images):
                VECT.append(LOG_P[i][0])
                Log_L+=LOG_P[i][1]
            print(Log_L)

            #break conditions
            if np.abs(Log_L - Log_L_old) < 0.1:
                print("result is :")
                print(priblij)
                break
            Log_L_old = Log_L

 ###################       M_step           #
            #weight
            vectors = np.array(VECT)
            Weigths=vectors.sum(0)/number_of_images
            print(Weigths)

            #mus
            priblij = np.zeros(shape)
            M_results = [pool.apply_async(M_step, (i, vectors[i],Weigths)) for i in range(number_of_images)]
            for el in M_results:
                priblij+=el.get()

            priblij = (priblij / number_of_images/number_of_angles)
            priblij = priblij.astype('int32')     #normalize IMG(float -> int32) can't make png from float

            print("current prediction is")
            print(priblij)

            im = Image.fromarray(priblij)
            im.save('Priblij{}.png'.format(num_of_iter))      #save img in png
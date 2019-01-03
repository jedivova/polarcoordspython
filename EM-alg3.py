import multiprocessing
import ctypes
import numpy as np
from PIL import Image
from scipy import ndimage
from math import sqrt, exp, log, pi, e
import matplotlib.pyplot as plt
import time
import os
from sys import platform
from decimal import Decimal
from My_polar5 import reproject_image_into_polar, polar_rotate

#arr = multiprocessing.Array(ctypes.c_int8, 2 * 1 * 3)
CPUnum=os.cpu_count()
term = 10
number_of_images = 1000
number_of_angles = int(360 / term)
sigma = 10

if platform == 'win32':
    shape = np.array(Image.open('imgs\img0.png')).shape
    way_to_img='imgs\img{}.png'
elif platform== 'linux':
    shape = np.array(Image.open('imgs/img0.png')).shape
    way_to_img = 'imgs/img{}.png'

from decimal import *
getcontext()
getcontext().Emin=-99999999
getcontext().prec = 3

def WeightsG(stepen,weigths):
    X=[]
    Len=stepen.shape[0]
    SUM=0
    #stepen -= stepen.max()  #i.e. *(e^x/e^x), where x is max degree
    for i in range(Len):
        X.append(Decimal(stepen[i]).exp()*Decimal(weigths[i]))
        SUM+=X[i]
    for i in range(Len):
        X[i] /= SUM

    arr=np.array(X)
    #print("sum of weigths", arr.sum())
    return arr, SUM


def LogL(priblij,img):
    a = ((priblij[img > 0] - img[img>0]) ** 2) /(2*sigma**2) # Можно пренебречь, т.к. сходится за то же кол-во итераций
    #a = ((priblij[img > 0] - img[img>0]) ** 2)
    logL = -a.sum()
    return logL



def E_step(number_of_image, priblijenie, weigths):
    #print('calculating log(L) for {} picture'.format(number_of_image))
    IMG = np.array(Image.open(way_to_img.format(number_of_image)))
    step = np.zeros(number_of_angles)
    for i in range(number_of_angles):
        angle = i * term
        REF = polar_rotate(priblijenie, -angle)
        step[i] = LogL(REF, IMG)
    #print(step)
    vector, SUM = WeightsG(step,weigths)
    #if number_of_image==3:
    #    print(weigths)
    #    print("{} IMG:".format(number_of_image), vector)
    #   print(SUM)

    return vector, SUM



def M_step(number_of_image,vector,weights):
    #print('calculating pixels for {} picture {}'.format(number_of_image,vector))
    #IMG = np.array(Image.open('rot{}.png'.format(number_of_image)).convert('RGB'))

    IMG_arr = np.array(Image.open(way_to_img.format(number_of_image)))
    priblij = np.zeros(IMG_arr.shape)
    for i in range(number_of_angles):
        angle = i * term
        #print('angle = {}'.format(angle))
        #REF = ndimage.rotate(IMG, -angle, reshape=False)
        IMG = polar_rotate(IMG_arr, angle)



        #a = np.float64(vector[i]/weights[i])
        a = np.float64(vector[i])
        priblij += a * IMG
        #print(f"{vector[i]} * {REF[55,55]} = {priblij[55,55]}")

    #plt.imshow(priblij, cmap=plt.cm.gray)
    #plt.savefig("Mstep_{}.png".format(number_of_image))

    return priblij


def M_STEP(vectors):
    a = [number_of_angles]
    a.extend(shape)

    priblij = np.zeros(a)
    dividers = np.zeros(a)

    # M_results = [pool.apply_async(M_step, (i, vectors[i],Weigths)) for i in range(number_of_images)]
    # for el in M_results:
    #    priblij+=el.get()

    for j in range(number_of_images):
        priblij_tmp, divider_tmp = M_step_Matrix_calc(vectors[j], j, a)

        priblij += priblij_tmp
        dividers += divider_tmp  # it is m in the formulae of EM
    dividers[dividers == 0] = 1
    priblij /= dividers
    priblij = priblij.sum(axis=0)
    ### Saving image ###
    priblij = priblij.astype('int32')  # normalize IMG
    im = Image.fromarray(priblij)
    if im.mode != 'L':
        im = im.convert('L')
    im.save('Priblij{}.png'.format(num_of_iter))  # save img in png

    return priblij


def M_step_Matrix_calc(vect,number_of_img, a):
    IMG_arr = np.array(Image.open(way_to_img.format(number_of_img)))

    argmax = vect.argmax()
    new_vect = np.roll(vect,-argmax)
    matrix = np.zeros(a)
    for i in range(number_of_angles):
        angle = term * (i+argmax)
        IMG = polar_rotate(IMG_arr, angle)
        matrix[i] = IMG
    matrix= matrix*np.float64(new_vect[:, None, None]) #multiply by g_ij     IMG* * g_ij
    divider = (matrix != 0).astype(int)  # true\false to 1\0

    return matrix, divider




if __name__ == '__main__':

    a=np.zeros((100,50))
    b=np.ones((100,50))*5
    priblij = np.concatenate((a,b), axis=1)
    #priblij = np.array(Image.open('imgs\img0.png'))
    #priblij = np.zeros(shape)
    Weigths = np.ones((number_of_angles)) / number_of_angles

    SUM_LOG1=0
    for num_of_iter in range(5):
        print("iter {}".format(num_of_iter))
        with multiprocessing.Pool(processes=CPUnum) as pool:
            #E_step
            time_start=time.time()
            E_results = [pool.apply_async(E_step, (i, priblij, Weigths)) for i in range(number_of_images)]
            LOG_P = [(el.get()) for el in E_results]
            #print(LOG_P)
            time_of_execution = time.time() - time_start
            print("E_step time: ", time_of_execution)

            VECT=[]
            SUM_LOG=0
            for i in range(number_of_images):
                VECT.append(LOG_P[i][0])
                #print(LOG_P[i][1])
                SUM_LOG+=LOG_P[i][1].ln()
            print("SUM_LOG = ",SUM_LOG)
            #print("VECT = ",VECT)

            ###################       M_step           #
            # weight
            vectors = np.array(VECT)
            #print(vectors[4])

            Weigths = vectors.sum(0) / number_of_images
            #print(Weigths)
            #print(Weigths.sum())

            priblij = M_STEP(vectors)

            #if SUM_LOG1==SUM_LOG:
            #    break
            SUM_LOG1=SUM_LOG
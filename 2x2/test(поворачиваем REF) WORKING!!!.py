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
    return arr

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

    vector = WeightsG(step,weigths)
    return vector


def Maximisation(g,dots):
    #weights
    G = g.sum(0)
    w = G/len(DOTS)

    #mu
    mu = np.zeros(k)
    for i in range(len(g)):
        mu += dots[i] * g[i]
    for j in range(k):
        mu[j]=mu[j]/len(DOTS)/w[j]

    #sigma
    sig = np.zeros(k)
    for j in range(k):
        for i in range(len(g)):
            sig[j] += (mu[j]-dots[i])**2 * g[i][j]
    for j in range(k):
        sig[j] = sig[j]/len(DOTS)/w[j]

    Guess = {'mu': mu,
             'sig': sig,
             'weights': w
             }

    return Guess


def M_step(number_of_image,vector,weights):
    #print('calculating pixels for {} picture {}'.format(number_of_image,vector))
    #IMG = np.array(Image.open('rot{}.png'.format(number_of_image)).convert('RGB'))

    IMG_arr = np.array(Image.open('2x2_{}.png'.format(number_of_image)))
    #priblij = np.zeros(shape)
    priblij = np.zeros((2,2))
    for i in range(number_of_angles):
        angle = i * term
        #print('angle = {}'.format(angle))
        #REF = ndimage.rotate(IMG, -angle, reshape=False)

        IMG = np.rot90(IMG_arr, i)

        priblij += vector[i] * IMG/weights[i]
        #print(f"{vector[i]} * {REF[55,55]} = {priblij[55,55]}")
    #plt.imshow(priblij)
    #plt.show()
    return priblij


def Number(j,numb_max_vect,w_max_number):
    numb = j - numb_max_vect + w_max_number
    if numb < 0:
        numb = numb + number_of_angles
    if numb>=number_of_angles:
        numb=numb-number_of_angles
    return numb


def Vectors(vector, m):
    #n=(vector.max(0)+vector.min(0))/2
    #print(vector)
    n = vector.max(0)
    vector -= n
    #print(n)
    #print(vector)

    #vector = e ** (vector)

    sum = 0
    for j in range(number_of_angles):
        #w_j = W[Number(j,vector.argmax(0),W.argmax(0))]
        #vector[j,z]=vector[j,z]*w_j
        sum += (vector[j])
    # print(sum)
    for j in range(number_of_angles):
        vector[j] = (vector[j]) / sum
    return vector


def Stepen(ss):
    uu = 0
    for u in range(100):

        ss = ss / 10
        if ss < 10:
            break
        uu += 1
    return uu


if __name__ == '__main__':
    #priblij = np.array(Image.open('2x2_0.png'))
    priblij = np.array([6, 0, 30, 4]).reshape((2, 2))
    print('priblij')
    print(priblij)
    print(type(priblij[0,0]))
    Weigths = np.ones((number_of_angles)) / number_of_images

    SUM_LOG1=0
    for num_of_iter in range(20):
        print("iter {}".format(num_of_iter))
        with multiprocessing.Pool(processes=CPUnum) as pool:
            #E_step
            E_results = [pool.apply_async(E_step, (i, priblij, Weigths)) for i in range(number_of_images)]
            LOG_P = [(el.get()) for el in E_results]
            print(LOG_P)


            SUM_LOG = 0
            for number_of_image in range(number_of_images):
                for j in range(number_of_angles):
                    SUM_LOG += LOG_P[number_of_image][j]
            SUM_LOG1=SUM_LOG
            print("SUM_LOG ",SUM_LOG1)
            #if abs(SUM_LOG1-SUM_LOG).max()<1:
            #    num_of_iter=30
             #   break
 ###################       M_step           #
            #weight
            vectors = np.copy(LOG_P)
            Weigths=vectors.sum(0)/number_of_images
            print(Weigths)


            priblij = np.zeros(shape)
            M_results = [pool.apply_async(M_step, (i, vectors[i],Weigths)) for i in range(number_of_images)]
            for el in M_results:
                priblij+=el.get()

            #print(priblij[50,50], priblij[60,60])
            #priblij = (priblij / number_of_images)
            priblij = (priblij / number_of_images/number_of_angles)
            print(priblij)
            print(type(priblij[0, 0]))
            priblij = priblij.astype('int32')     #normalize IMG
            print("after normalization", type(priblij[0, 0]))
            #print(priblij[50, 50], priblij[60, 60])
            #np.save('Image', priblij)   #save in arr type
            im = Image.fromarray(priblij)
            im.save('Priblij{}.png'.format(num_of_iter))      #save img in png
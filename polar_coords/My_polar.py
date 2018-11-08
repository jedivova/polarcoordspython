import pandas as pd
import numpy as np

def Get_polar_coords(IMG):
    # IMG = np.arange(16).reshape((4,4))
    ny, nx = IMG.shape
    origin_x, origin_y = nx // 2, ny // 2
    A = np.arange(nx)
    B = np.arange(ny)

    if nx % 2 != 0:
        A -= origin_x
        x_max = origin_x
        print("X_max",x_max )
    else:
        A[0: origin_x] -= origin_x
        A[origin_x:] -= origin_x - 1
        x_max = origin_x
        print("X_MAX",x_max )

    if ny % 2 != 0:
        B -= origin_y
    else:
        B[0: origin_y] -= origin_y
        B[origin_y:] -= origin_y - 1

    x_coords, y_coords = np.meshgrid(A, B)

    r = np.sqrt(x_coords ** 2 + y_coords ** 2)  # add /2 for easing calculations
    theta = np.arctan2(y_coords, x_coords)
    #print(r)
    #print(np.degrees(theta))
    if ny % 2 != 0:
        theta_min = 0
    else:
        theta_min = np.arctan2(1, nx // 2)
    theta_max = np.arctan2(-1, nx // 2) + np.pi * 2
    #theta_max = np.pi*2-theta_min
    print("theta_mim-max", theta_min, theta_max)
    if ny % 2 == 0 and nx % 2 == 0:
        min_radius = np.sqrt(2)
    elif ny % 2 == 0 or nx % 2 == 0:
        min_radius = 1
    else:
        min_radius = 0
    max_radius = np.sqrt((IMG.shape[0] // 2) ** 2 + (IMG.shape[1] // 2) ** 2)
    print("max_rad",min_radius, max_radius)
    #print(IMG.shape)
    print(IMG)
    #print(x_coords)
    #print(y_coords)
    print(r)
    #print(r.shape)
    print(theta)

    return r, theta


def IMG_to_polar(IMG):
    r, phi = Get_polar_coords(IMG)
    columns=['X','Y', 'PIX', 'radius', 'Phi']
    rows=[]
    shape = IMG.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            rows.append([i, j, IMG[i,j], r[i,j], phi[i,j]])

    df = pd.DataFrame(rows, columns=columns)

    return df


def _under_pi():
    return 1


def Rotating_polar(Data, degree):
    if len(Data)%2 != 0:
        N=int(len(Data)/2)

        Data['Phi'] += np.radians(degree)

        Data.loc[Data.Phi > np.pi, 'Phi'] -= 2*np.pi
        Data.loc[Data.Phi < -np.pi, 'Phi'] += 2 * np.pi
        Data.loc[N,'Phi'] = 0
    else:
        Data['Phi'] += np.radians(degree)
        Data.loc[Data.Phi > np.pi, 'Phi'] -= 2*np.pi
        Data.loc[Data.Phi < -np.pi, 'Phi'] += 2 * np.pi

    return Data


def polar_to_cart(df):      # NOW WORKING
    nx, ny = int(df.max()['X'])+1, int(df.max()['Y'])+1
    origin_x, origin_y = nx // 2, ny // 2
    IMG=np.zeros((nx, ny))
    for i in range(len(df)):
        x, y, PIX, r, phi = df.iloc[i]
        y = int(round(r*np.cos(phi)))+origin_x
        if x>0:
            pass
        x = int(round(r*np.sin(phi)))+origin_y
        IMG[x,y] = PIX

    return IMG


#IMG = np.arange(16).reshape((4,4))
#IMG = np.arange(4).reshape((2,2))
#IMG = np.arange(9).reshape((3,3))
#IMG = np.arange(12).reshape((4,3))
IMG = np.arange(12).reshape((3,4))

max_radius = np.sqrt((IMG.shape[0]//2)**2+(IMG.shape[1]//2)**2)
print(max_radius)
df = IMG_to_polar(IMG)
print(df)


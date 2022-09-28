import math
import random
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from PIL import Image

p = [151,160,137,91,90,15,
   131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
   190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
   88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
   77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
   102,143,54, 65,25,63,161,1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
   135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
   5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
   223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
   129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
   251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
   49,192,214, 31,181,199,106,157,184,84,204,176,115,121,50,45,127, 4,150,254,
   138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180, 151,160,137,91,90,15,
   131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
   190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
   88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
   77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
   102,143,54, 65,25,63,161,1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
   135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
   5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
   223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
   129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
   251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
   49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
   138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180]

g = [(1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),
(1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1),
(0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1)]

def dot3d(a, b):
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])

def lin_interp(v, a, b):
    return a + v * (b - a)

def perlin3d(x, y, z):

    xcell = math.floor(x)
    ycell = math.floor(y)
    zcell = math.floor(z)
    
    xf = x % 1
    yf = y % 1
    zf = z % 1

    boBottomLeftDist = [xf,yf,zf]
    boBottomRightDist = [xf-1,yf,zf]
    boTopLeftDist = [xf,yf-1,zf]
    boTopRightDist = [xf-1,yf-1,zf]
    topBottomLeftDist = [xf,yf,zf-1]
    topBottomRightDist = [xf-1,yf,zf-1]
    topTopLeftDist = [xf,yf-1,zf-1]
    topTopRightDist = [xf-1,yf-1,zf-1]

    boBottomLeftGrad = g[p[p[p[xcell] + ycell] + zcell] % 12]
    boBottomRightGrad = g[p[p[p[xcell + 1] + ycell] + zcell] % 12]
    boTopLeftGrad = g[p[p[p[xcell] + ycell + 1] + zcell] % 12]
    boTopRightGrad = g[p[p[p[xcell + 1] + ycell + 1] + zcell] % 12]
    topBottomLeftGrad = g[p[p[p[xcell] + ycell] + zcell + 1] % 12]
    topBottomRightGrad = g[p[p[p[xcell + 1] + ycell] + zcell + 1] % 12]
    topTopLeftGrad = g[p[p[p[xcell] + ycell + 1] + zcell + 1] % 12]
    topTopRightGrad = g[p[p[p[xcell + 1] + ycell + 1] + zcell + 1] % 12]

    boBottomLeftdot = dot3d(boBottomLeftDist, boBottomLeftGrad)
    boBottomRightdot = dot3d(boBottomRightDist, boBottomRightGrad)
    boTopLeftdot = dot3d(boTopLeftDist, boTopLeftGrad)
    boTopRightdot = dot3d(boTopRightGrad, boTopRightDist)
    topBottomLeftdot = dot3d(topBottomLeftGrad, topBottomLeftDist)
    topBottomRightdot = dot3d(topBottomRightDist, topBottomRightGrad)
    topTopLeftdot = dot3d(topTopLeftDist, topTopLeftGrad)
    topTopRightdot = dot3d(topTopRightGrad, topTopRightDist)
   
    u = 6*xf*xf*xf*xf*xf - 15*xf*xf*xf*xf + 10*xf*xf*xf
    v = 6*yf*yf*yf*yf*yf - 15*yf*yf*yf*yf + 10*yf*yf*yf
    c = 6*zf*zf*zf*zf*zf - 15*zf*zf*zf*zf + 10*zf*zf*zf

    return lin_interp(c, lin_interp(u, lin_interp(v, boBottomLeftdot, boTopLeftdot), lin_interp(v, boBottomRightdot, boTopRightdot)),
     lin_interp(u, lin_interp(v, topBottomLeftdot, topTopLeftdot), lin_interp(v, topBottomRightdot, topTopRightdot)))





def animatedPerlinSphere():

    ''' 3d Perlin noise projected on a sphere, animated by altering influence and frequency values.
        WARNING - This will save the GIF, and all individual frames to the current directory.
    '''

    frames = 367
    influence = 0
    frequency = 1

    for n in range(frames):
        plt.clf()
        fig = plt.figure() 
        frame = fig.add_subplot(111, projection="3d")
        influence += 0.1
        frequency += 0.02
    
        sphere_center = (150, 150, 150)
        D = 50
        lin_x = []
        lin_y = []
        lin_z = []
        colourmap = []
        for row in range(360):
            beta = row * 0.01
            lin_z.append([])
            lin_x.append([])
            lin_y.append([])
            colourmap.append([])
            for column in range(360):
                theta = column
                x = (D * math.sin(beta) * math.cos(theta)) + sphere_center[0]
                y = (D * math.sin(beta) * math.sin(theta)) + sphere_center[1]
                z = (D * math.cos(beta) + sphere_center[2])
                noise = perlin3d(x / frequency, y / frequency, z / frequency) * influence
                lin_x[row].append(x + noise)
                lin_y[row].append(y + noise)
                lin_z[row].append(z + noise)
                colourmap[row].append(abs(noise))
        lin_x = np.array(lin_x)
        lin_y = np.array(lin_y)
        lin_z = np.array(lin_z)
        colourmap = np.array(colourmap)
        minn, maxx = colourmap.min(), colourmap.max()
        norm = matplotlib.colors.Normalize(minn, maxx)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap='jet')
        m.set_array([])
        fcolors = m.to_rgba(colourmap)
        frame.plot_surface(lin_x,lin_y,lin_z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
        plt.savefig(f"{n}.png")
    images = [Image.open(f"{n}.png") for n in range(frames)]

    images[0].save('ball.gif', save_all=True, append_images=images[1:], duration=math.floor(367/40), loop=0)

def perlinSphere():
    ''' Non animated 3d Perlin noise projected on a sphere - visual stretching due to 3d matplotlib '''
    sphere_center = (150, 150, 150)
    D = 50
    lin_x = []
    lin_y = []
    lin_z = []
    colourmap = []
    influence = 5
    frequency = 5
    for row in range(360):
        beta = row * 0.01
        lin_z.append([])
        lin_x.append([])
        lin_y.append([])
        colourmap.append([])
        for column in range(360):
            theta = column
            x = (D * math.sin(beta) * math.cos(theta)) + sphere_center[0]
            y = (D * math.sin(beta) * math.sin(theta)) + sphere_center[1]
            z = (D * math.cos(beta) + sphere_center[2])
            noise = perlin3d(x / frequency, y / frequency, z / frequency) * influence
            lin_x[row].append(x + noise)
            lin_y[row].append(y + noise)
            lin_z[row].append(z + noise)
            colourmap[row].append(abs(noise))

    lin_x = np.array(lin_x)
    lin_y = np.array(lin_y)
    lin_z = np.array(lin_z)
    colourmap = np.array(colourmap)

    # Create custom colormap
    minn, maxx = colourmap.min(), colourmap.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap='inferno')
    m.set_array([])
    fcolors = m.to_rgba(colourmap)

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(lin_x,lin_y,lin_z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
    plt.show()

def threeDimensionalPlane():
        ''' 2d Perlin noise on a 3d plane '''
        z = 0
        shape = (250, 250)
        noise = np.zeros(shape)
        for xa in range(shape[0]):
            row=[]
            x = xa * 0.01
            for ya in range(shape[1]):
                y = ya * 0.01
                noise[xa][ya] = perlin3d(x, y, z)
        lin_x = np.linspace(0, 1, shape[0], endpoint=False)
        lin_y = np.linspace(0, 1, shape[1], endpoint=False)
        a, b = np.meshgrid(lin_x, lin_y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(a,b,noise,cmap='gray')
        plt.show()

# DEMO

perlinSphere()

threeDimensionalPlane()

# ONLY RUN IF YOU ARE WILLING TO SAVE INDIVIDUAL FRAMES

# animatedPerlinSphere()


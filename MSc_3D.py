# encoding=utf-8
# code by
# import
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi,sin,cos,arccos
import itertools
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import cv2
from PIL import Image

# **********************************************
#      user Defined parameter 用户自定义变量
# **********************************************
faceNumber = np.random.uniform(25, 30)
figNumber = 2
volume = 10

# **********************************************
#                  function
# **********************************************
#
def cacu_Radius(volume):
    radius = (volume*3/(np.pi*4))**(1/3)
    # print(f"Radius of base:{radius}")
    return radius

# randomly generate vertex
def createVertices(radius=1, faceNumber=4):
    '''
    :param verticeNumber:number of vertices
    :return: coordinates of vertices
    '''
    verticeNumber = int((faceNumber-4)/2)+4
    def checkVertex(radius, vertex, vertices):
        # vertex -- randomly generate vertec coord
        # vertices -- selected vertices coord
        sign = True
        for point in vertices:
            x1, y1, z1 = point[0], point[1], point[2]
            x2, y2, z2 = vertex[0], vertex[1], vertex[2]
            distance = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
            if distance<0.3:
                sign = False
                break
        return sign

    vertices = []  # store vertex coord
    while True:
        angle1 = np.random.uniform(0, pi*2)
        angle2 = np.random.uniform(0, pi*2)
        # randomly generate x,y,z coord
        z = radius*cos(angle1)
        x = radius*sin(angle1)*cos(angle2)
        y = radius*sin(angle1)*sin(angle2)
        if len(vertices):
            if checkVertex(radius, [x, y, z], vertices):
                vertices.append([x, y, z])
            else:
                pass
        else:
            vertices.append([x, y, z])

        if len(vertices) >= verticeNumber:
            break
    return vertices

# calculate normal vector
def nmlVector(coords):
    coord1, coord2, coord3 = coords[0],coords[1],coords[2]
    x1, y1, z1 = coord1[0], coord1[1], coord1[2]
    x2, y2, z2 = coord2[0], coord2[1], coord2[2]
    x3, y3, z3 = coord3[0], coord3[1], coord3[2]

    a = (y2-y1)*(z3-z1) - (y3-y1)*(z2-z1)
    b = (z2-z1)*(x3-x1) - (z3-z1)*(x2-x1)
    c = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    n = np.array([a, b, c])
    n = n/np.linalg.norm(n)
    d = arccos(np.dot(n, coord1)/np.linalg.norm(coord1))
    if d>pi/2:  # keep normal vector outward
        n = -n
    return n

# surface check
def validSurface(vertices):
    vertices = [np.array(vertice) for vertice in vertices]
    threePoints = list(itertools.combinations(vertices, 3))
    chosenSurface = []
    for threePoint in threePoints:
        n = nmlVector(threePoint)
        sign = True
        for vertex in vertices:
            vector1 = threePoint[0]-vertex
            result = np.dot(vector1, n)
            if result<-1e-5:
                sign = False
                break
        if sign:
            chosenSurface.append(list(threePoint))
    return chosenSurface

# Draw and save
def drawPicture(draw, figN=0, thred=1):
    # Render 3D shapes
    ax = a3.Axes3D(pl.figure(figsize=(3, 3), dpi=600))
    for i in range(len(draw)):
        vtx = draw[i]
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_color(colors.rgb2hex([1, 1, 1]))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    # change axis
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    #
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')

    plt.gca().set_box_aspect((1, 1, 1))
    # Save image
    plt.savefig("./3d_aggregate-{}.png".format(figN))
    # plt.show()
    # load image - Gaussian Blur
    image = cv2.imread("./3d_aggregate-{}.png".format(figN))
    # result = cv2.GaussianBlur(image, (120, 100), 75)
    result = cv2.GaussianBlur(image, (45, 45), 15)
    cv2.imwrite("./3d_aggregate-{}-Gauss.png".format(figN), result)

    # set Th - modify image

    imageNum = np.array(Image.open("./3d_aggregate-{}-Gauss.png".format(figN)).convert("L"))

    imageNum[np.where(imageNum > thred)] = 255
    imageNum[np.where(imageNum <= thred)] = 0

    plt.figure()
    plt.imshow(imageNum, cmap='gray')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    # plt.show()
    plt.savefig("./3d_aggregate-{}-thred.png".format(figN))


# calculate volume
def calVolume(chosenSurface):
    volume = 0.0
    def volume_single(points):
        a, b, c = points[0], points[1], points[2]
        v1 = abs(np.dot(np.cross(a, b), c))/6
        return v1

    for points in chosenSurface:
        volume += volume_single(points)

    return volume

# renew surface coord
def changeVertices(chosenSurface, user_volume):
    chosenSurface = np.array(chosenSurface)
    upper, lower = 5, 0
    while True:
        center = (upper+lower)/2.
        changeSurface = chosenSurface * center
        volume = calVolume(changeSurface)
        if volume < user_volume:
            lower = center
        else:
            upper = center

        if abs(volume-user_volume)/user_volume < 0.001:
            print(f"center={center},volume={volume}")
            break

    return changeSurface


# **********************************************
#                  Main code
# **********************************************
thred = 215    # np.random.uniform(0, 255)
for i in range(figNumber):
    radius = cacu_Radius(volume)
    vertices = createVertices(radius=radius, faceNumber=faceNumber)
    surface = validSurface(vertices)
    changeSurface = changeVertices(chosenSurface=surface, user_volume=volume)
    drawPicture(changeSurface, i,  thred)





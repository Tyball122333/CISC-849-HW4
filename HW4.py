import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from statistics import mean

%matplotlib inline

dim1 = 90
dim2 = 120
K = 4
R_THRESH = 175
G_THRESH = 175
B_THRESH = 150

VEC_Y = np.array([255,255,0])
COLORS = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255], 4: [
    255, 255, 0], 5: [255, 0, 255], 6: [0, 255, 255], 7: [255, 255, 255], 8: [128, 128, 128]}
C_NAMES = ['Black', 'Red', 'Green', 'Blue', 'Yellow', 'Magneta', 'Cyan', 'White', 'Grey']

def addPlot(fig, img, pos, title):
    fig.add_subplot(1, 3, pos)
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)

def calcDist(vec):
    return np.linalg.norm(vec-VEC_Y)

for num in range(1,31):
    IMG_NAME = f'ball{num}.jpg'

    fig = plt.figure()
    img = cv2.imread(IMG_NAME)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (dim2, dim1))
    addPlot(fig, img, 1, f'Input({num})')

    data = []
    for i in range(len(img)):
        row = []
        for j in range(len(img[i])):
            if img[i][j][0] > R_THRESH and img[i][j][1] > G_THRESH and img[i][j][2] < B_THRESH:
                row.append(img[i][j])
            else:
                row.append([0, 0, 0])
        data.append(row)
    addPlot(fig, data, 2, 'Processed')

    data = np.reshape(data, (dim1*dim2, 3))
    kmeans = KMeans(n_clusters=K).fit(data)
    results = [COLORS[label] for label in kmeans.labels_]
    results = np.reshape(results, (dim1, dim2, 3))
    addPlot(fig, results, 3, 'Output')

    closest = min(kmeans.cluster_centers_, key=calcDist)
    coords = []
    ind = np.where(kmeans.cluster_centers_ == closest)[0][0]
    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i] == ind:
            col = i % dim2
            row = i // dim2
            coords.append((col,row))
    avgX = mean([x for (x,y) in coords])
    avgY = mean([y for (x,y) in coords])
    
    print(f'{num}: {C_NAMES[ind]}')
    print(f'({avgX},{avgY})')
    #minDist = calcDist(closest)
    #print(f'{num}:', closest)
    
    
#r = []
#g = []
#b = []
#for i in range(len(img)):
#    for j in range(len(img[i])):
#        r.append(img[i][j][0])
#        g.append(img[i][j][1])
#        b.append(img[i][j][2])
#
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter3D(r,g,b)
#ax.set_xlabel('r')
#ax.set_ylabel('g')
#ax.set_zlabel('b')
#plt.show()

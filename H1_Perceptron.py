import numpy as np
import matplotlib.pyplot as plt

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0 , 0 , 0 ,1]
Data = []
for k in range(len(x)):
    Data += [x[k] + [y[k]]]

print('Data is :')
print(Data)

x_coords = [point[0] for point in Data]
y_coords = [point[1] for point in Data]
z_status = [point[2] for point in Data]

for i in range(len(x)):
    if z_status[i] == 1:
        plt.scatter(x_coords[i], y_coords[i], color='blue', s=200, marker='o')
    else:
        plt.scatter(x_coords[i], y_coords[i], color='blue', s=200, marker='o', facecolors='none', lw=1.5)

plt.xlim(-1, 2)
plt.ylim(-1, 2)
plt.grid(True)

w =np.random.rand(3)

# add bias
Data = [[-1] + point for point in Data]
Data = np.array(Data)
print('Data after add bias :')
print(Data)

#learning rate
eta = 0.1

wStar = []
x_values = []

epoch = 10
endEpoce = epoch
for epoch in range(epoch):
    print(f'Epoce {epoch+1} started')
    e = 0
    np.random.permutation(Data)
    for j in range(len(Data)):
        x_values = np.linspace(-2, 2, 100)
        wStar = (-w[2] * x_values - w[0]*Data[j][0]) / w[1]
        Prediction = w[0] * Data[j][0] + w[1] * Data[j][1] + w[2] * Data[j][2]
        f = 1 if Prediction > 0 else 0
        error = Data[j][3] - f
        print(f'error: {error}')
        w = w + eta * error * Data[j][0:3]
        if error==0 : e +=1
    plt.plot(wStar,x_values, color=np.random.rand(3), lw=2,linestyle='--')
    print(f'updated weights : {w}')
    print(f'Epoce {epoch+1} ended')

    #stopping criterion
    if e == len(Data) :
        print(f" network could converged to the answer in {epoch+1} epoch ")
        plt.plot(wStar, x_values, color='red', lw=3)
        break
    if epoch+1 == endEpoce and e != len(Data) :
        print(f" network couldn't converged to the answer in {epoch+1} epoch ")

plt.show()
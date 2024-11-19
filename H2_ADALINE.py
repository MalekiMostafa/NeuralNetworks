import numpy as np
import matplotlib.pyplot as plt

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [-1 , -1 , -1 ,1 ]
Data = []
for k in range(len(x)):
    Data += [x[k] + [y[k]]]

print('Data is :')
print(Data)


x_coords = [point[0] for point in Data]
y_coords = [point[1] for point in Data]
z_status = [point[2] for point in Data]


for i in range(len(Data)):
    if z_status[i] == 1:
        plt.scatter(x_coords[i], y_coords[i], color='blue', s=50, marker='o')
    else:
        plt.scatter(x_coords[i], y_coords[i], color='red', s=50, marker='o', facecolors='none', lw=1.5)

plt.xlim(-1,2)
plt.ylim(-1,2)
plt.grid(True)



w =np.zeros(3)

Data = [[1] + point for point in Data]
Data = np.array(Data)

print('Data after add bias :')
print(Data)

#learning rate
eta = .05

wStar = []
x_values = []
errors_per_epoch = []

epoch = 100
endEpoce = epoch
error_threshold = 1e-3
PreviousError = 0

#leatning rate scheduler

for epoch in range(epoch):
    print(f'Epoce {epoch+1} started')

    np.random.permutation(Data)
    TotalError = 0
    for j in range(len(Data)):
        # Prediction = np.dot(w, Data[j][0:3])
        Prediction = w[0] * Data[j][0] + w[1] * Data[j][1] + w[2] * Data[j][2]
        print(f'Prediction: {Prediction}')
        #print(f'Prediction is {Prediction}')
        error = Data[j][3] - Prediction
        print(f'error: {error}')
        TotalError += error*error

        w = w + eta * error * Data[j][0:3]

    errors_per_epoch.append(TotalError)

    print(f'updated weights : {w}')
    print(f'Epoce {epoch + 1} ended')
    print('')
    if abs(TotalError - PreviousError) < error_threshold:
        #print(f" TotalError - PreviousError is {TotalError - PreviousError} ")
        x_values = np.linspace(-2, 2, 100)
        wStar = (-w[0]*Data[j][0] - w[1] * x_values) / w[2]
        print(f"          network converged in {epoch + 1} epochs ")
        plt.plot( x_values, wStar,  color='red', lw=2 )
        plt.title('Data Separating Line')
        break

    if epoch + 1 == endEpoce :
        print(f" network couldn't converged in {epoch + 1} epochs ")
        x_values = np.linspace(-2, 2, 100)
        wStar = (-w[0]*Data[j][0] - w[1] * x_values) / w[2]
        plt.plot(x_values,wStar, color=np.random.rand(3), lw=2, linestyle='--')
        plt.title('The last line obtained')
        break

    PreviousError=TotalError

plt.grid(True)
plt.show()

plt.figure()
plt.plot(range(1, len(errors_per_epoch) + 1), errors_per_epoch, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Total Error')
plt.title('Error per Epoch')
plt.grid(True)

plt.show()
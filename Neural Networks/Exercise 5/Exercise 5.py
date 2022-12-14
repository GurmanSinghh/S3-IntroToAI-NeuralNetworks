#!/usr/bin/env python
# coding: utf-8

# In[10]:




import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl



'''Exercise # 5: Three input multi-layer feed forward to recognize sum pattern with more training data'''


''' #5.1 Repeat exercises # 1 but instead of having two inputs generate three inputs'''

#Set the seed and generate three arrays of random numbers
np.random.seed(1)
set1 = np.random.uniform(-0.6, 0.6, 10).reshape(10,1)
set2 = np.random.uniform(-0.6, 0.6,10).reshape(10,1)
set3 = np.random.uniform(-0.6, 0.6, 10).reshape(10,1)

#Concatenate the three samples
input_gurman = np.concatenate((set1, set2, set3), axis=1)

#Inspect the concatenated array
print(input_gurman)
print(input_gurman.shape)

#Calculate the target variable based on the three inputs
target = (input_gurman[:, 0] + input_gurman[:, 1] + input_gurman[:, 2]).reshape(10,1)

print(target)

# Minimum and maximum values for each dimension
dim1_min, dim1_max = input_gurman[:,0].min(), input_gurman[:,0].max()
dim2_min, dim2_max = input_gurman[:,1].min(), input_gurman[:,1].max()
dim3_min, dim3_max = input_gurman[:,2].min(), input_gurman[:,2].max()


# Define a single-layer neural network
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
dim3 = [dim3_min, dim3_max]
nn = nl.net.newff([dim1, dim2, dim3], [6,1])

# Train the neural network
error = nn.train(input_gurman, target, show=15, goal=0.00001)

print(f"Number of inputs: {nn.ci}")
print(f"Number of outputs: {nn.co}")
print(f"Number of layers, including hidden layer: {len(nn.layers)}")


plt.plot(error)
plt.xlabel('Epoch number')
plt.ylabel('Train error')
plt.grid()
plt.show()


result5 = nn.sim([[0.2, 0.1, 0.2]])

print(f'Result 5: {result5}')



''' #5.2 Repeat exercise #4 but instead of having two inputs generate three inputs'''

#Set the seed and generate three arrays of random numbers
np.random.seed(1)
set1 = np.random.uniform(-0.6, 0.6, 100).reshape(100,1)
set2 = np.random.uniform(-0.6, 0.6, 100).reshape(100,1)
set3 = np.random.uniform(-0.6, 0.6, 100).reshape(100,1)

#Concatenate the three samples
input_gurman = np.concatenate((set1, set2, set3), axis=1)

print(input_gurman)
print(input_gurman.shape)

#Calculate the target variable based on the three inputs
target = (input_gurman[:, 0] + input_gurman[:, 1] + input_gurman[:, 2]).reshape(100,1)

print(target)

# Minimum and maximum values for each dimension
dim1_min, dim1_max = input_gurman[:,0].min(), input_gurman[:,0].max()
dim2_min, dim2_max = input_gurman[:,1].min(), input_gurman[:,1].max()
dim3_min, dim3_max = input_gurman[:,2].min(), input_gurman[:,2].max()



# Define a single-layer neural network
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
dim3 = [dim3_min, dim3_max]

nn = nl.net.newff([dim1, dim2, dim3], [5,3,1])

# Set the training algorithm to gradient descent
nn.trainf = nl.train.train_gd

# Train the neural network
error = nn.train(input_gurman, target, epochs=1000, show=100, goal=0.00001)

print(f"Number of inputs: {nn.ci}")
print(f"Number of outputs: {nn.co}")
print(f"Number of layers, including hidden layers: {len(nn.layers)}")


plt.plot(error)
plt.xlabel('Epoch number')
plt.ylabel('Train error')
plt.grid()
plt.show()

result6 = nn.sim([[0.2, 0.1, 0.2]])

print(f'Result 6: {result6}')


# In[ ]:





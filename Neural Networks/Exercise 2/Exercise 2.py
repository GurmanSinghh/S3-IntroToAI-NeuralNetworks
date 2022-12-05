#!/usr/bin/env python
# coding: utf-8

# In[6]:




import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl



'''Exercise # 2: Multi-layer feed forward to recognize sum pattern'''

#Set the seed and generate two arrays of random numbers
np.random.seed(1)
set1 = np.random.uniform(-0.6, 0.6, 10).reshape(10,1)
set2 = np.random.uniform(-0.6, 0.6, 10).reshape(10,1)

#Concatenate the two samples
input_gurman = np.concatenate((set1, set2), axis=1)

#Inspect the concatenated array
print(input_gurman)
print(input_gurman.shape)

#Calculate the target variable based on the two inputs
target = (input_gurman[:, 0] + input_gurman[:, 1]).reshape(10,1)

print(target)

# Minimum and maximum values for each dimension
dim1_min, dim1_max = input_gurman[:,0].min(), input_gurman[:,0].max()
dim2_min, dim2_max = input_gurman[:,1].min(), input_gurman[:,1].max()


# Define a single-layer neural network with 2 hidden layers. One with 5 and the other with 3 neurons
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]

nn = nl.net.newff([dim1, dim2], [5,3,1])

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


#Predict a value with one single test sample
result2 = nn.sim([[0.1,0.2]])

print(f'Result 2: {result2}')


# In[ ]:





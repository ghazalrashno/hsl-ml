import numpy as np
from LinearRegrassionMulti import LinearRegrassionMulti
from pylab import show, title, xlabel, ylabel, plot
LRM = LinearRegrassionMulti()

print('Loading data ...')

# Load Data

data = np.loadtxt('ex1data2.txt', delimiter=',')
datax = data[:, 0]
datay = data[:, 2]
m = len(datay)  # number of training examples
datax = data[:, 0:2].reshape((m, 2))
datay = data[:, 2].reshape((m, 1))

print('xshape = ', datax.shape)
print('yshape = ', datay.shape)

# Print out some data points

print('First 10 examples from the dataset: ')
print(' x10 = [', data[0:10, :], ']', 'y10 = [', datay[0:10, :], ']')
input('Program paused. Press enter to continue.')

# Scale features and set them to zero mean

print('datax shape = ', datax.shape)
print('Normalizing Features ')

# plot(list(range(m)), datax[:, 1])

datax, mu, sigma = LRM.featureNormalize(datax)

print('mu = ', mu)
print('sigma = ', sigma)
print('datax shape = ', datax.shape)

# plot(list(range(m)), datax[:, 1])
# show()

# Add intercept term to X

datax = np.c_[np.ones(m), datax]

# ================Part 2: Gradient Descent================

print('Running gradient descent ')

# Choose some alpha value

alpha = 0.0001
num_iters = 15000

# Init Theta and Run Gradient Descent

theta = np.zeros((3, 1))
theta, J_history = LRM.gradientDescentMulti(
    datax, datay, theta, alpha, num_iters)

print('J_history = ', J_history)
print('J_history_size = ', J_history.size)

J_list = list(range(J_history.size))
# Plot the convergence graph
plot(J_list, J_history)
title('minimizing cost function')
xlabel('Number of iterations')
ylabel('Cost J')
show()

print('Theta computed from gradient descent: ')
print(theta)

sqrft = 1650
numbed = 3
theta0 = theta[0, :]
theta1 = theta[1, :]
theta2 = theta[2, :]

print('theta0 test', theta0, theta1, theta2)

price = theta0+theta1*(sqrft)+theta2*(numbed)

print('estimatd price', price)

# ================Part 3: Normal Equations================

print('press inter to continue')
print('Solving with normal equations')

# Load Data

dataeq = np.loadtxt('ex1data2.txt', delimiter=',')
dataxeq = dataeq[:, 0]
datayeq = dataeq[:, 2]
m = len(datayeq)  # number of training examples
dataxeq = dataeq[:, 0:2].reshape((m, 2))
datayeq = dataeq[:, 2].reshape((m, 1))

print('xshape = ', dataxeq.shape)
print('yshape = ', datayeq.shape)

# Add intercept term to X

dataxeq = np.c_[np.ones(m), dataxeq]

# Calculate the parameters from the normal equation

thetaeq = LRM.normalEqn(dataxeq, datayeq)

# Display normal equation's result

print('Theta computed from the normal equations:', thetaeq)

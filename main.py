
import matplotlib.pyplot as plt
from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour, legend
from LinearRegression import LinearRegression
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

LR = LinearRegression()


def surf(Z, X, Y):

    C = Z

    scalarMap = cm.ScalarMappable(norm=Normalize(
        vmin=C.min(), vmax=C.max()), cmap=cm.jet)

    # outputs an array where each C value is replaced with a corresponding color value
    C_colored = scalarMap.to_rgba(C)

    fig = plt.figure()

    ax = Axes3D(fig)

    surf = ax.plot_surface(X, Y, Z, facecolors=C_colored)

    return surf


# ====================Part 1: Basic Function====================

print('Running warmUpExercise ...')
print('5x5 Identity Matrix:')
print(LR.warmupex())

input('Program paused. Press enter to continue.')

# ========================Part 2: Plotting======================

print('plotting data')
data = np.loadtxt('ex1data1.txt', delimiter=',')
datax = data[:, 0]
datay = data[:, 1]
m = len(datay)  # number of training examples
datax = data[:, 0].reshape((m, 1))
datay = data[:, 1].reshape((m, 1))
# LR.plotdata(datax, datay)

input('Program paused. Press enter to continue.')

# =============Part 3: Cost and Gradient descent================

# Add a column of ones to x

one = np.ones((m, 1))
datax = np.append(one, datax, axis=1)
print(datax)
print(datax.shape)
# initialize fitting parameters

theta = np.zeros((2, 1))

# Some gradient descent settings
alpha = 0.01
num_iters = 1500
print('Testing the cost function ...')

# compute and display initial cost

J = LR.computecost(datax, datay, theta)
print('With theta = [0 ; 0]Cost computed =', J)
print('Expected cost value (approx) 32.07')

# further testing of the cost function

thetatest = np.array([[-1], [2]])
J = LR.computecost(datax, datay, thetatest)
print('With theta = [-1 ; 2]Cost computed = ', J)
print('Expected cost value (approx) 54.24')

input('Program paused. Press enter to continue.')

# run gradient descent

print('Running Gradient Descent ...')
theta, J_history = LR.gradientdescent(datax, datay, theta, alpha, num_iters)
# plot(list(range(num_iters)), J_history)
# show()
# print theta to screen

print('Theta found by gradient descent:')
print('  ', theta)
print('Expected theta values (approx)')
print(' -3.6303  1.1664')

# Plot the linear fit
# hold on; % keep previous plot visible

scatter(datax[:, 1], datax.dot(theta))
xlabel('Training data')
ylabel('Linear regression')
show()

# Predict values for population sizes of 35, 000 and 70, 000
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of ', predict1*10000)
predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of ', predict2*10000)

input('Program paused. Press enter to continue.')

# =============Part 4: Visualizing J(theta_0, theta_1)=============

print('Visualizing J(theta_0, theta_1) ')

# Grid over which we will calculate J

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's

J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))

# Fill out J_vals

for i in range(0, len(theta0_vals)):

    for j in range(0, len(theta1_vals)):
        t = [[theta0_vals[i]], [theta1_vals[j]]]
        J_vals[i, j] = LR.computecost(datax, datay, t)

print(J_vals)

X, Y = np.meshgrid(theta0_vals, theta1_vals)

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped

J_vals = np.transpose(J_vals)

# Surface plot
surf(X=X, Y=Y, Z=J_vals)
fig2 = plt.figure(figsize=(6, 5))
# Contour plot

input('press enter ')

contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
xlabel('theta_0')
ylabel('theta_1')
scatter(theta[0], theta[1])
show()

input('press inter ')

legend(loc='lower right')
show()
print(theta)

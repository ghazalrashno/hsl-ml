# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import numpy as np


class LinearRegression:

    def warmupex(self):

        eye = np.identity(5)

        return eye

    def plotdata(self, datax, datay):

        scatter(x=datax, y=datay, marker='o', c='b')
        title('Profits distribution')
        xlabel('Population of City in 10,000s')
        ylabel('Profit in $10,000s')
        show()

    def computecost(self, datax, datay, theta):

        m = len(datay)
        predictions = datax.dot(theta)
        cost = (1/(2*m)) * np.sum(np.square(predictions - datay))

        return cost

    def gradientdescent(self, datax, datay, theta, alpha, num_iters):

        m = len(datay)
        J_history = np.zeros((num_iters, 1))

        for i in range(num_iters):

            prediction = np.dot(datax, theta)
            theta = theta - (1/m) * alpha * (datax.T.dot((prediction-datay)))
            J_history[i] = self.computecost(datax, datay, theta)

        return theta, J_history

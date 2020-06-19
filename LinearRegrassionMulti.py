import numpy as np


class LinearRegrassionMulti:

    def computeCostMulti(self, datax, datay, theta):

        m = len(datay)
        predictions = datax.dot(theta)
        cost = (1/(2*m)) * np.sum(np.square(predictions - datay))

        return cost

    def gradientDescentMulti(self, datax, datay, theta, alpha, num_iters):

        m = len(datay)
        J_history = np.zeros((num_iters, 1))

        # for i in range(num_iters):

        #     error = datax.dot(theta) - datay
        #     theta = theta - (alpha/m) * (datax.T.dot(error))
        #     J_history[i] = self.computeCostMulti(datax, datay, theta)
        # return theta, J_history
        for i in range(num_iters):

            prediction = np.dot(datax, theta)
            theta = theta - (1/m) * alpha * (datax.T.dot((prediction-datay)))
            J_history[i] = self.computeCostMulti(datax, datay, theta)

        return theta, J_history

    def normalEqn(self, datax, datay):

        theta = np.zeros((np.size(datax, 1), 1))
        xpinv = np.linalg.pinv(datax.T.dot(datax))
        x = datax.T.dot(datay)
        theta = xpinv.dot(x)

        return theta

    def featureNormalize(self, datax):

        X_norm = datax
        mu = np.zeros((1, np.size(datax, 1)))
        sigma = np.zeros((1, np.size(datax, 1)))
        mu = np.mean(datax)
        sigma = np.std(datax)
        X_norm = (datax - mu)/sigma

        return X_norm, mu, sigma

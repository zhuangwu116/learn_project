# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np

import random

def gradientDescent(x, y, theta, alpha, m, numIterations):
    
    xTrans = x.transpose()

    for i in range(0, numIterations):
        
        hypothesis = np.dot(x, theta)

        loss = hypothesis - y

        cost = np.sum(loss ** 2) / (2 * m)

        print("Iteration %d / Cost: %f" % (i, cost))

        gradient = np.dot(xTrans, loss) / m

        theta = theta - alpha * gradient

    return theta

def genData(numPoints, bias, variance):

    x = np.zeros(shape = (numPoints, 2))

    y = np.zeros(shape = numPoints)

    for i in range(0, numPoints):

        x[i][0] = 1

        x[i][1] = i




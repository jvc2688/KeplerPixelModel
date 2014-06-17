import numpy as np

def leastSquareSolve(a, y, l2=0):
#250000.0

#using normal equation
    fy = np.dot(a.T, y)
    fa = np.dot(a.T, a)
#adding l2 regularization
    i = np.identity(fa.shape[0])
    re = l2 * i
    result = []
    result.append(np.linalg.solve(fa+re, fy))
#calculating the residual and rms deviation
    fit = np.dot(a, result[0])
    res = y - fit
    resSq = np.sum(res**2, axis=0)
    result.append(resSq)
    ratio = np.divide(y, fit)
    dev = ratio - 1.0
    rms = np.sqrt(np.mean(dev**2, axis=0))
    result.append(rms)
    return result
'''
    
#using svd
    eps = np.finfo(float).eps
    n = a.shape[0]
    u, s, v = np.linalg.svd(a)
    coe = np.zeros((a.shape[1],y.shape[1]))
    max = np.amax(s)
    for i in range(s.shape[0]):
        if s[i] <= eps:
            pass
        else:
            for j in range(coe.shape[1]):
                coe[:,j] += s[i]*np.dot(u[:,i].T, y[:,j])/(s[i]**2+l2)*v.T[:,i]
    result = []
    result.append(coe)
    fit = np.dot(a, result[0])
    res = y - fit
    resSq = np.sum(res**2, axis=0)
    result.append(resSq)
    ratio = np.divide(y, fit)
    dev = ratio - 1.0
    rms = np.sqrt(np.mean(dev**2, axis=0))
    result.append(rms)
    return result
'''
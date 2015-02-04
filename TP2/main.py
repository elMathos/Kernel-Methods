# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 22:11:22 2015

@author: Clement Nicolle
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels

X = np.loadtxt("zip.train")
# dim 257, first line is the digit, we don't need it
# we only keep the first 1000 points

X = X[:, 1:]
# let's plot a digit to see :
plt.imshow(X[0].reshape((16, 16)), cmap=plt.cm.gray)

# Kernel PCA

# 1- linear kernel
kpca_lin = KernelPCA(kernel="linear")
X_kpca_lin = kpca_lin.fit_transform(X)
# 2- polynomial kernel
kpca_poly = KernelPCA(kernel="poly")
X_kpca_poly = kpca_poly.fit_transform(X)
# 3- rbf kernel
kpca_rbf = KernelPCA(kernel="rbf")
X_kpca_rbf = kpca_rbf.fit_transform(X)

# Plot results for first 200 points:
nb_points = 200
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1, aspect='equal')
plt.scatter(X_kpca_lin[:nb_points, 0], X_kpca_lin[:nb_points, 1])
plt.title("Linear kernel")
plt.subplot(1, 3, 2, aspect='equal')
plt.scatter(X_kpca_poly[:nb_points, 0], X_kpca_poly[:nb_points, 1])
plt.title("Polynomial kernel")
plt.subplot(1, 3, 3, aspect='equal')
plt.scatter(X_kpca_rbf[:nb_points, 0], X_kpca_rbf[:nb_points, 1])
plt.title("Gaussian kernel")
plt.show()

# Denoising

# add Gaussian noise :
noise = np.random.normal(0, 0.2, 256)
noisy_img = X[0] + noise
plt.imshow(noisy_img.reshape((16, 16)), cmap=plt.cm.gray)


def compute_gamma(y, X, d, std_dev):
    n_samples = len(X)
    kpca = KernelPCA(kernel="rbf", gamma=1/(2*std_dev**2))
    kpca.fit(X)
    dim = len(kpca.lambdas_)  # dim is smaller than n_samples, weird

    # compute K on dataset
    K = np.ones((n_samples, n_samples))
    K_noisy = np.ones(n_samples)
    gamma = np.zeros(dim)
    for i in range(n_samples):
        K_noisy[i] = np.exp(-np.linalg.norm(X[i]-y)**2/(2*std_dev**2))
        for j in range(i+1, n_samples):
            K[i, j] = np.exp(-np.linalg.norm(X[i]-X[j])**2/(2*std_dev**2))
            K[j, i] = K[i, j]

    for i in range(dim):
        for j in range(d):
            for k in range(dim):
                gamma[i] += kpca.alphas_[j,k]*kpca.alphas_[j,i]*(K_noisy[j]-np.sum(K[:,k])/dim)/kpca.lambdas_[j]
        print str(i)
    return (gamma+1./dim)


X_short = X[:200, :]
gam = compute_gamma(noisy_img, X_short, 50, 10)





std_dev = 10
def function_to_optim(y):
    s = 0
    for i in range(199): 
        s -= gam[i]*np.exp(-np.linalg.norm(X[i]-y)**2/(2*std_dev**2))
    return s

test = function_to_optim(noisy_img)

import scipy.optimize as opt
u = opt.minimize(function_to_optim, x0=X[3])
denoised = u["x"]
plt.imshow(denoised.reshape((16, 16)), cmap=plt.cm.gray)





def denoising_gamma(gamma, X, n_iter):
    n_samples = len(X)
    n_feat = len(X[0])
    dim = len(gamma)
    y = X[np.random.randint(0, n_samples)]  # initialization
    L_y = [y]
    for i in range(n_iter):
        K_img = pairwise_kernels(X, y, metric="rbf")
        K_img = K_img.reshape(n_samples)
        numer = np.zeros(n_feat)
        denom = 0
        for j in range(dim):
            numer += gamma[j]*K_img[j]*X[j]
            denom += gamma[j]*K_img[j]
        y = numer/denom
        L_y . append(y)

    return L_y
    

def denoising(noisy_img, X, d, std_dev, n_iter):
    gamma = compute_gamma(noisy_img, X, d, std_dev)
    return denoising_gamma(gamma, X, std_dev, n_iter)

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
# we only keep the first 200 points
X = X[:1000, 1:]
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
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1, aspect='equal')
plt.scatter(X_kpca_lin[:200, 0], X_kpca_lin[:200, 1])
plt.title("Linear kernel")
plt.subplot(1, 3, 2, aspect='equal')
plt.scatter(X_kpca_poly[:200, 0], X_kpca_poly[:200, 1])
plt.title("Polynomial kernel")
plt.subplot(1, 3, 3, aspect='equal')
plt.scatter(X_kpca_rbf[:200, 0], X_kpca_rbf[:200, 1])
plt.title("Gaussian kernel")
plt.show()

# Denoising

# add Gaussian noise :
noise = np.random.normal(0, 0.4, 256)
noisy_img = X[0] + noise
plt.imshow(noisy_img.reshape((16, 16)), cmap=plt.cm.gray)


def compute_gamma(noisy_img, X, d, std_dev):
    n_samples = len(X)
    kpca = KernelPCA(kernel="rbf", gamma=std_dev)
    kpca.fit(X)
    dim = len(kpca.lambdas_)

    # compute gamma_i coefs
    params = {"gamma": std_dev}
    K = pairwise_kernels(X, metric="rbf", **params)
    K_noisy = pairwise_kernels(X, noisy_img, metric="rbf", **params)
    H = np.eye(n_samples) - np.ones((n_samples, n_samples))  # center matrix
    K_noisy_c = np.dot(H, K_noisy.reshape(n_samples)
                       - np.dot(K, np.ones(n_samples))/n_samples)
    alpha = np.zeros(dim)
    for i in range(dim):
        for k in range(d):
            s = 0
            for j in range(dim):
                s += kpca.alphas_[k, j]*kpca.alphas_[k, i]*K_noisy_c[j]
            alpha[i] += s
        alpha[i] /= kpca.lambdas_[i]
    gamma = alpha + (1-sum(alpha))/n_samples

    return gamma


def denoising_gamma(gamma, X, std_dev, n_iter):
    n_samples = len(X)
    n_feat = len(X[0])
    dim = len(gamma)
    params = {"gamma": std_dev}
    y = X[np.random.randint(0, n_samples)]  # initialization
    L_y = [y]
    for i in range(n_iter):
        K_img = pairwise_kernels(X, y, metric="rbf", **params)
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

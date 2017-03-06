#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:59:02 2016

@author: gazal_chawla, manasa_anil, vaishnavi
"""
# set dictionarySize in method computeImageSIFT, pathImageDir in method main to the desired values

import time
import cv2
import glob
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
import sklearn.cluster
from sklearn import metrics
from itertools import cycle


# method to output the current time
def currentTime():
    localtime = time.asctime(time.localtime(time.time()))
    return localtime

# method to plot clustering output
def plotClusters(X, cluster_centers_indices, labels, n_clusters_):
    plt.close('all')
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        for x in X[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

# method to calculate silhouette score
def calculateSilhouette(X, labels, affinityVal):
    return metrics.silhouette_score(X, labels, metric=affinityVal)

# method to plot a numpy 2D array
def plotArray(X):
    plt.scatter(X[:,0], X[:,1])
    plt.show()

# method to generate SIFT features of images in a directory with some extension
def computeImageSIFT(pathImageDir, extension):
    dictionarySize = 100 # image dictionary size is set to the desired value
    BOW = cv2.BOWKMeansTrainer(dictionarySize)
    for file in glob.glob(join(pathImageDir, extension)):
        img = cv2.imread(join(pathImageDir, file), 0)
        if img is not None:
            sift = cv2.xfeatures2d.SIFT_create()
            kp, desc = sift.detectAndCompute(img, None)
            BOW.add(desc)
    return BOW

# method to compute BOW Dictionary
def computeBOWDictionary(BOW):
    dictionary = BOW.cluster()
    sift2 = cv2.xfeatures2d.SIFT_create()
    BOWDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
    BOWDiction.setVocabulary(dictionary)
    return BOWDiction

# method to compute BOW descriptors for images
def computeImageBOW(pathImageDir, extension, BOWDiction):
    imgDesc = []
    for file in glob.glob(join(pathImageDir, extension)):
        img = cv2.imread(join(pathImageDir, file), 0)
        if img is not None:
            sift = cv2.xfeatures2d.SIFT_create()
            des = BOWDiction.compute(img, sift.detect(img))
            imgDesc.append(des[0])
    return np.asarray(imgDesc)


# method to perform hierarchical clustering; returns labels and centroids
def performHierarchical(X, noCluster, affinityVal, linkageVal):
    ac = sklearn.cluster.AgglomerativeClustering(n_clusters=noCluster, affinity=affinityVal, linkage=linkageVal).fit_predict(X)
    #extracting centroids of hierarchical clusters
    codebook = []
    for i in range(ac.min(), ac.max()+1):
        codebook.append(X[ac == i].mean(0))
    centroid = np.vstack(codebook)
    return ac, centroid

# method to perform AP Clustering
def performAP(X, dampingVal, convergeVal, maxVal):
    af = AffinityPropagation(damping=dampingVal, convergence_iter=convergeVal, max_iter=maxVal).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    return cluster_centers_indices, labels, n_clusters_

# method that calls all the other methods
def main():
   print "Start at :", currentTime()
   pathImageDir = '/Users/gazalchawla/Desktop/untitled folder 2' # path of the directory that contains the images
   extension = '*.JPEG' # extension of the images
   BOW = computeImageSIFT(pathImageDir, extension)
   print "SIFT Descriptors generated: ", currentTime()
   BOWDictionary = computeBOWDictionary(BOW)
   print "BOW Dictionary generated: ", currentTime()
   desArray = computeImageBOW(pathImageDir, extension, BOWDictionary)
   noCluster = 50 # number of clusters in the hierarchical clustering
   affinityVal = 'euclidean'
   linkageVal = 'average'
   labelH, centroidH = performHierarchical(desArray, noCluster, affinityVal, linkageVal)
   print "Hierarchical Clustering performed: ", currentTime()
   dampingVal = 0.9
   convergeVal = 200
   maxVal = 2000
   cluster_centers_indices, labels, n_clusters_ = performAP(centroidH, dampingVal, convergeVal, maxVal)
   print "AP Clustering performed: ", currentTime()
   print "Points before hierarchical clustering was performed: "
   plotArray(desArray)
   print "Points after hierarchical clustering was performed: "
   plotArray(centroidH)
   print "Points in final clusters after AP clustering was performed: "
   plotClusters(centroidH, cluster_centers_indices, labels, n_clusters_)
   print "Silhouette Score for clustering: ", calculateSilhouette(centroidH, labels, affinityVal)
   print "Ends at :", currentTime()

main()

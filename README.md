# ITCS-6157-Automatic-Image-Clustering

Program Statement:
Download 1 million images and extract SIFT features of each. Apply AP clustering technique to get an optimum partition of the images according to their visual similarity.


Algorithm Steps:

I. Obtain the set of bags of features

• Select a large set of images.

• Extract the SIFT feature points of all the images in the set and obtain the SIFT descriptor
for each feature point that is extracted from each image.

• Cluster the set of feature descriptors for the amount of bags we defined and train the bags
with clustered feature descriptors (we use the K-Means algorithm).

• Obtain the visual dictionary.

II. Obtain the Bag of Words descriptor for given image

• Extract SIFT feature points of the given image.

• Obtain SIFT descriptor for each feature point.

• Match the feature descriptors with the dictionary we created in the first step.

• Build the histogram.

III. Obtain the points representing 1M images

• Perform hierarchical clustering on the image dataset and obtain a fixed number of
centroids.

• These centroids will represent the image dataset.

IV. Obtain the clusters in the dataset

• The centroids after hierarchical clustering are given as input to AP clustering algorithm.

• The final number of clusters are obtained thereafter.

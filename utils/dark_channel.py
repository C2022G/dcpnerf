import os.path

import cv2
import numpy as np
from sklearn.cluster import KMeans


def Atomospheric_light_k_means(image):
    # Compute dark channel prior
    kernel_size = 15
    dark_channel = cv2.erode(cv2.dilate(image.min(axis=-1), np.ones((kernel_size, kernel_size))),
                             np.ones((kernel_size, kernel_size)))

    # Estimate atmospheric light intensity using cluster statistics
    candidate_points = np.argwhere(dark_channel >= np.percentile(dark_channel, 99.9))
    kmeans = KMeans(n_clusters=5).fit(candidate_points)
    clusters = [candidate_points[kmeans.labels_ == i] for i in range(5)]
    clusters.sort(key=lambda x: len(x), reverse=True)
    atmospheric_light = np.mean([image[p[0], p[1]] for p in clusters[0]], axis=0)
    return atmospheric_light

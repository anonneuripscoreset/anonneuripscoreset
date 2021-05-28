# Contains code for maximin sampling and voronoi weighting. 
from collections import Counter
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
import numpy as np

def maximin_sample(data, num_indices):
    """
    Returns a maximin sample.
    """

    # Start can be the first point or a random one. Does not really matter.
    starting_index = random.randint(0, len(data) - 1)
    # starting_index = 0
    mList = []
    latest_sampled_index = starting_index
    distance_list = cdist(data, [data[starting_index]])
    distance_list = np.linalg.norm(data - data[starting_index], axis = 1)
    for t in range(num_indices):
        newest_distances = np.linalg.norm(data - data[latest_sampled_index], axis = 1)
        distance_list = np.minimum(newest_distances, distance_list)
        latest_sampled_index = np.argmax(distance_list) 
        mList.append(latest_sampled_index)
    return data[mList]

def get_voronoi_weights(data, samples):
    """
    Given some samples, weigh each prototype according to how many points
    in the big data set get assigned to it.
    """
    D = pairwise_distances(data, samples)
    closest = np.argmin(D, axis = 1)
    c = Counter(closest)
    ls = []
    for i in range(len(samples)):
        if i in c:
            ls.append(c[i])
        else:
            ls.append(0)
    return ls
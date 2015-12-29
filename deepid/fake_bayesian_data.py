import numpy as np
import pickle

clusters_number = 4
features_number = 4
points_per_cluster = 20
std_dev = np.random.uniform(size=[clusters_number])

data = [[] for i in range(clusters_number)]
for i in range(clusters_number):
    for j in range(points_per_cluster):
        one_hot = np.zeros([features_number])
        one_hot[i] = 10
        noise = np.random.normal(0, std_dev[i], features_number)
        data[i].append(one_hot+noise)


def center(matrix):
    mean = np.zeros([features_number])
    observations = 0
    for element in matrix:
        for subelement in element:
            mean += subelement
            observations += 1
    mean /= observations
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = [matrix[i][j][k]-mean[k]
                            for k in range(features_number)]
    return matrix

new_data = center(data)
pickle.dump(new_data, open('fake_data_centered', 'w'))

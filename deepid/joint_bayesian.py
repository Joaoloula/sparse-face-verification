# An implementation of the joint bayesian face verification algorithm
# from PIL import Image
from __future__ import division
import numpy as np
import itertools
import pickle

features_number = 4
people_number = 530
threshold = 0.5
# train_set = pickle.load(open('ordered_test', 'rb'))
train_set = pickle.load(open('fake_data_centered', 'rb'))


def make_pairs(dataset):
    labeled_list = [[photo, i] for i in range(len(dataset))
                    for photo in dataset[i]]
    pairs = itertools.combinations(labeled_list, 2)
    return pairs


class joint_bayesian:
    # Defines the joint bayesian class, which contains the Su and Se objects, as
    # well as methods for initializing them with random gaussians and training

    def __init__(self, dimension):
        # Initialize with Su and Se as square matrices of the dimension of the
        # feature space.
        self.Su = np.random.normal(0, 0.5, [dimension, dimension])
        self.Se = np.random.normal(0, 0.5, [dimension, dimension])

    def train(self, dataset):
        # Train the model on images of one person, previous Su and Se
        # estimations and the number of features n
        u = np.array([])
        e = np.array([])
        # F and G (that will be calculated once for loop) are two matrices that
        # will help us compute the expected value of u and e more efficiently
        F = np.linalg.pinv(self.Se)
        for personal_images in dataset:
            m = len(personal_images)  # Number of images
            G = -np.linalg.pinv((m+1)*self.Su+self.Se).dot(self.Su).dot(F)
            # We find the expected values of u and e
            x = np.add.reduce(personal_images)
            u = np.append(u, self.Su.dot(F+(m+1)*G).dot(x))
            for j in range(m):
                e = np.append(e, np.dot(self.Se, personal_images[j]) +
                              self.Se.dot(G).dot(x))
        # Update Su and Se based on the covariances of u and e
        u = np.reshape(u, [-1, features_number])
        e = np.reshape(e, [-1, features_number])
        self.Su = np.cov(u.T)
        self.Se = np.cov(e.T)

    def eval(self, dataset):
        # Make iterable object for the labeled pairs
        labeled_pairs = make_pairs(dataset)
        # Test model accuracy on a batch of labeled pairs (0=match, 1=mismatch)
        loss = 0
        # We'll use the matrix A to aid is in calculating the log likelihood
        A = (np.linalg.pinv(self.Su + self.Se) -
             np.linalg.pinv(
                            (self.Su + self.Se) -
                            self.Su
                            .dot(np.linalg.pinv(self.Su+self.Se))
                            .dot(self.Su)
                            )
             )
        # Calculation of the log-likelihood r as seen on the paper
        for pair in labeled_pairs:
            r = (np.transpose(pair[0][0]).dot(A).dot(pair[0][0]) +
                 np.transpose(pair[1][0]).dot(A).dot(pair[1][0]) -
                 2*np.transpose(pair[0][0]).dot(A).dot(pair[1][0]))

            if pair[0][1] == pair[1][1]:
                loss += 1/r
            else:
                loss += r
        print loss


model = joint_bayesian(features_number)

for i in range(10):
    model.train(train_set)
    model.eval(train_set)
# for identity in range(people_number):
#     person = train_set[identity]
#     batch = []
#     for image in person:
#         grayscale = np.asarray(Image.open(image).convert('1'))
#         batch.append(grayscale)
#     # Train the model for the given person
#     model.train(batch)

# An implementation of the joint bayesian face verification algorithm
# from PIL import Image
import numpy as np
import pickle

features_number = 160
people_number = 530
# train_set = pickle.load(open('ordered_test', 'rb'))
train_set = pickle.load(open('fake_data_centered', 'rb'))


def initialize(dimension):
    # Initializes a centered gaussian square matrix of the specified dimension
    return np.random.normal(0, 0.5, [dimension, dimension])


class joint_bayesian:
    # Defines the joint bayesian class, which contains the Su and Se objects, as
    # well as methods for initializing them with random gaussians and training
    def __init__(self, dimension):
        # Initialize with Su and Se as square matrices of the dimension of the
        # feature space.
        self.Su = initialize(dimension)
        self.Se = initialize(dimension)

    def train(self, personal_images):
        # Train the model on images of one person, previous Su and Se
        # estimations and the number of features n
        m = len(personal_images)  # Number of images
        print m
        # F and G are two matrices that will help us compute the expected value
        # of u and e more efficiently
        F = np.linalg.pinv(self.Se)
        print F
        G= -np.linalg.pinv((m+1)*self.Su + self.Se).dot(self.Su).dot(F)
        print G
        # We find the expected values of u and e
        x = np.add.reduce(personal_images)
        # u = np.zeros([features_number, people_number])
        u = self.Su.dot(F+(m+1)*G).dot(x)
        e = np.zeros([features_number, m])
        for j in range(m):
            e[:, j] = (np.dot(self.Se, personal_images[j]) +
                       self.Se.dot(G).dot(x))
        # Update Su and Se based on the covariances of u and e
        u_mean = np.add.reduce(u)
        u = u - u_mean
        u_cov = [u[i]*u[j] for i in range(features_number)
                           for j in range(features_number)]
        self.Su = np.reshape(u_cov, [160, 160])
        self.Se = np.cov(e.T, rowvar=0)

model = joint_bayesian(features_number)

for identity in train_set:
    model.train(identity)

# for identity in range(people_number):
#     person = train_set[identity]
#     batch = []
#     for image in person:
#         grayscale = np.asarray(Image.open(image).convert('1'))
#         batch.append(grayscale)
#     # Train the model for the given person
#     model.train(batch)

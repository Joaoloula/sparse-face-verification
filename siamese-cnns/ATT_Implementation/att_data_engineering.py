# import os
from PIL import Image
import numpy as np
import pickle
import itertools

faces_and_labels = []
for i in range(40):
    for j in range(10):
        faces_and_labels.append(
            [np.asarray(
                 Image.open(
                            '/home/loula/Programming/python/face_verification' +
                            '/siamese_cnns/att_faces/s' + str(i+1) + '/' +
                            str(j+1) + '.pgm'
                            )
                 .crop((10, 0, 92, 102)).resize((64, 64), Image.ANTIALIAS)),
             i+1]
        )

# Random version
pairs = list(itertools.combinations(faces_and_labels, 2))
np.random.shuffle(pairs)
[train_random, test_random] = [pairs[1:60001], pairs[60001:]]
pickle.dump([train_random, test_random], open('train_test_random', 'w'))

# Split version (siamese paper)
# train_faces = faces_and_labels[1:321]
# test_faces = faces_and_labels[321:361]
# train_split = list(itertools.combinations(train_faces, 2))
# test_split = list(itertools.combinations(test_faces, 2))
# pickle.dump([train_split, test_split], open('train_test_split', 'w'))

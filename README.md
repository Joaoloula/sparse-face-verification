## Verification ##
Face verification is a classic problem in computer vision: given two pictures representing each one a face, how to determine whether they belong to the same person or not? In this project we'll take a machine learning approach and focus specifically on the case where we want to test new inputs against some subset of people of whom we have few examples in the training data (for example, in biometrics applications). For this, we'll try two approaches:

## Exemplar SVMs##
Exemplar SVMs were first introduced in 2011 by Malisiewicz et al.[1] The idea is to train one linear SVM classifier for each exemplar in the training set, so that we end up with one positive instance and lots of negatives ones. Surprisingly, this very simple idea works really well, getting results close to the state of the art at the PASCAL VOC object classification dataset at the time. 

Here, starting from one sole picture of the person we wish to identify, we train an Exemplar SVM model to be able to do verification against the person in question. 

## Siamese CNNs##
The Siamese CNN architecure was first proposed by Yann LeCun, and has a long history of use in face verification [2][3][4]. The idea is to train two identical CNNs that share parameters, and whose outputs are fed to an energy function that will measure how "dissimilar" they are, upon which we'll then compute our loss function. Gradient descent on this loss propagates to the two CNNs in the same way, preserving the symmetry of the problem.

We'll experiment with Siamese CNNs first on the AT&T face dataset (https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) for proof of concept and to find a model that's robust to comparison of faces it has never seen before. We'll then move on to the more challenging Labeled Faces in the Wild dataset (http://vis-www.cs.umass.edu/lfw/), where we'll deal with all the problems that arise from uneven positions, lighting and background. 

[1] https://www.cs.cmu.edu/~tmalisie/projects/iccv11/exemplarsvm-iccv11.pdf

[2] http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf

[3] http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6903759

[4] http://arxiv.org/pdf/1406.4773v1.pdf

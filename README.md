# Sparse Face Verification #

Face verification is a classic problem in computer vision: given two pictures representing each one a face, how to determine whether they belong to the same person or not? In this project we'll take a machine learning approach and focus specifically on the case where we want to test new inputs against some subset of people of whom we have few examples in the training data (for example, in biometrics applications).

## Exemplar SVMs ##
Our first approach is based on a method first introduced in 2011 by Malisiewicz et al.[1]: Exemplar SVMs. The idea is to train one linear SVM classifier for each exemplar in the training set, so that we end up with one positive instance and lots of negatives ones. Surprisingly, this very simple idea works really well, getting results close to the state of the art at the PASCAL VOC object classification dataset at the time. 

First, we're gonna run our training set through a HOG descriptor. HOG (Histogram of Oriented Gradients) descriptors are a nifty feature descriptor based on gradient/edge detection. The idea is to divide the image into cells, in which all the pixels will "vote" for the preferred gradient by means of an histogram (the weight of each pixel's vote is proportional to the gradient magnitude). The resulting set of histograms is the descriptor, and it's been proven to be robust to many kinds of large-scale transformations and thus widely used in object detection (you can learn more about HOG descriptors in the great INRIA article [here](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)).

The next step is to fit a linear SVM model *for each positive example in the dataset*. These SVMs will take as input only that positive example and the thousands of negative ones, and try to find the hyperplane that maximizes the margin between them in the HOG feature space. The next step is to bring all these exemplar SVMs together by means of a *calibration*, in which we rescale our space using the validation set so that the best examples get pulled closer to our positive -- and the worst ones, further apart (without altering their order!).  From there we can compare these images between exemplar SVMs, compute a score for each one, and decide on a threshold upon which to make our decision.

<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/sparse-face-verification/master/images/calibration.jpg"/>
  <span> Visualization of an exemplar SVM before and after calibration. </span>
</p>

This way of sewing together the results from the different SVMs reduces our problems with overfitting, the elephant in the room whenever we're talking about learning from few examples.

With our implementation, depending on the person chosen, the model can get a bit more than 80% accuracy on [LFW dataset](http://vis-www.cs.umass.edu/lfw/), which seems really good for such a simple algorithm. This result, however, is based on false assumptions: by cropping LFW using face recognition and alignment to remove background, performance falls drastically. This seems to be a common issue on LFW: our model is overfitting to the background, as it's very susceptible to do so.

<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/sparse-face-verification/master/images/bush.jpg"/>
  <span> Example of pair that overfits by background: there are many groups of images like this on LFW, which explains how, by overfitting to most of the possible backgrounds for each person, we managed to get such good results. </span>
</p>

## Siamese CNNs ##
So, that didn't work as well as we hoped. Let's think harder about our problem then: what we're trying to do is find a representation of our data such that intrapersonal distances (differente pictures of the same person) are small and interpersonal distances (pictures of different people) are big. In the linear case, this can be thought of as finding a symmetric positive definite matrix M such that the distance defined by:

<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/sparse-face-verification/master/images/Mahalanobis.jpg"/>
</p>

This metric is called a Mahalanobis distance, and it's an object of great interest in statistics. The matrix M is the object we're interested in learning. The characterization of M allows us to write it as a product of another matrix W and its transpose, and so, using some linear algebra, we find that the Mahalanobis distance is equivalent to:

<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/sparse-face-verification/master/images/Mahalanobis_Alternative.jpg"/>
</p>

That is, the Euclidean distance calculated between the application of W to our two input vectors.

We're gonna borrow the main idea from this equivalent Mahalanobis metric learning problem, namely finding a transformation W such that the associated distance has the properties of low intrapersonal distance and high interpersonal distance. We are not, however, going to limit our search to the linear transformation space: face data has a very complicated structure, and it's mostly thought that it lies upon a high-dimensional manifold: we'll be in trouble if we try to navigate through it linearly. How, then, can we learn a non-linear transformation that captures the subtleties of this space all the while preserving the symmetry of the problem? That's where siamese convolutional neural networks come in.

The siamese CNN architecure was first proposed by Chopra and LeCun, and has a long history of use in face verification [2][3][4]. The idea is to train two identical CNNs that share parameters, and whose outputs are fed to an energy function that will measure how "dissimilar" they are, upon which we'll then compute our loss function. Gradient descent on this loss propagates to the two CNNs in the same way, preserving the symmetry of the problem. Notice that, by choosing our energy function to be the euclidean distance, we find ourselves with exactly the setup we described above.

We'll experiment with siamese CNNs first on the [AT&T face dataset](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) with a simple 4-layer architecture for proof of concept and to find a model that's robust to comparison of faces it has never seen before. We'll then move on to the more challenging LFW, where we'll deal with all the problems that arise from uneven positions, lighting and background. 

While we do manage to get a jump in accuracy, our model suffers from the simplicity of its architecture, and doesn't seem very robust to the introduction of faces it has never seen on the dataset.

## DeepID ##

That was better, but we're not quite there yet. In mathematics, when facing a hard problem, it sometimes helps to consider a way harder problem that can give us some insight into how to proceed. Well, we can think of face verification as a subproblem of face *identification*, that is, the classification problem that involves assigning to each person a label: their identity. In the case of face verification, we're just trying to know if this assignment is the same for two given points in our dataset. 

The jump from verification to identification can certainly be quite troublesome: in our earlier example of biometrics, to prevent the entrance of undesired people, the owner of the system would ideally have to train his algorithm to recognize all seven billion people on earth! Far from this naïve approach, however, lies an interesting connection that makes the exploration of this harder problem worthwhile for us: both problems are based on the recognition of facial features, so we'd hope that training a neural network to perform the hard problem of identification would give us very good descriptors for verification. That is the core idea behind DeepID, a state-of-the-art algorithm for face verification.

DeepID made quite the fuss on CVPR14 [5] by getting better benchmarks on LFW than DeepFace, Facebook team's method that had by far the best results ever seen on the challenging dataset: the apparently crazy idea of trying to predict 10,000 classes for the same number of different people worked incredibly well.

This still leaves us with the problem of what algorithm to use for the verification task after removing the final layer on the CNN. With great methods comes great responsibility, and so we'll choose something fancy so as to get the best out of DeepID: we'll implement what's called a joint-bayesian model.

Joint-bayesian models operate on our earlier framing of inter and intra-class distances. What we do now, however, is suppose that the class centers mu as well as the intra class variations epsilon follow the both of them a centered gaussian distribution, whose parameters we'll try to infer from the data. 

DeepID is not made for training on LFW: there are too few examples per person to achieve satisfactory results. Instead, we are going to use the [FaceScrub dataset](http://vintage.winklerbros.net/facescrub.html), that consists of 100,000 images of about 500 people. By training our CNN architecture to recognize these people, we'll get descriptors that we'll then apply to the verification problem on LFW.

[1] https://www.cs.cmu.edu/~tmalisie/projects/iccv11/exemplarsvm-iccv11.pdf

[2] http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf

[3] http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6903759

[4] http://arxiv.org/pdf/1406.4773v1.pdf

[5] http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf

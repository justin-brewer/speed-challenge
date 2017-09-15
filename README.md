README

About :

  This project was started in response to comma.ai's speed challenge:
  http://commachallenge.s3-us-west-2.amazonaws.com/speed_challenge_2017.tar
  Predict the speed of a moving vehicle using only its dash cam footage and
  a labeled training set.

Approach :

  Calculate dense optical flow with OpenCV's Farneback implementation.
  Shrink the data using spline interpolation.
  Fully-connected, feedforward network with Tensorflow: four hidden layers -
  3 x relu + softmax. The output layer's weights are not trainable and I used
  mean squared error loss (1). Data is shuffled before being split into
  train and test sets. Bootstrap aggregation is used to pick the batch at each
  training step. 50% dropout was used during training.

Results :

  Training on 50%, my network achieves MSE of 1.8 ((m/s)^2) on my validation
  set.
  Training on 60%, MSE = 1.2.
  Training on 70%, MSE = 0.06.

  A large portion of the error seems to come from times the car is sitting
  still and other cars are moving in the camera's view. More training data is
  needed to know if this model is actually useful.
  ...to be continued.

Future Work :

  Other than averaging optical flow over a window of five flow frames
  (which consists of information from six video frames), this network is memoryless.
  I believe, adding some form of recurrence to the network will boost performance
  significantly. I also plan to experiment with using a CNN in conjunction with
  the current network. I originally tried using a CNN on the optic flow data,
  but with poor results, which makes sense, assuming the correlation between
  optic flow and velocity are not spatially invariant. I also think it would be
  interesting to try to calculate the optical flow using a CNN (2). The softmax layer
  needs to be extended to support speed predictions higher than the max speed in the
  training set. Experiment with data preparation methods before training.

Files:

  save_opflow.py :

    Dense Optical flow, calculated with OpensCV's implementation
    of the Farneback Algorithm: 'cv2.calcOpticalFlowFarneback()'.
    It is then shrunk by a factor of 16 with Scipy's Zoom, which uses
    spline interpolation. Then vectorized and saved in a .npy file.

  avg_data.py : (deprecated)

    Average data over some window size to smooth noise. I got good
    results with a window size of 5, but am no longer using it.

  c2_one_hot.py : (deprecated)

    Convert speeds to one-hot with a certain precision. The original
    network used a softmax output with cross-entropy loss, but now
    it uses the softmax layer as a hidden layer.

  network.py :

    Tensorflow model with four hidden layers : 3 x relu + softmax
    hidden and final layers weights are not learnable (1).
    Trains model, then saves it as .ckpt. The network still loads one hot
    labels to arrange the hidden softmax layer, obviously inefficient,
    will change soon.

  load_saved.py :

    Loads saved models by name, runs them on new data, then saves
    the model's predictions to a .npy file

  show_vid.py :

    Shows videos with speed predictions in m/s and mph.

* Note : lines in files where file names go are marked with '#37'

Dependencies/tools:
    numpy==1.13.1
    opencv-python==3.3.0.9
    scipy==0.19.1
    tensorflow==1.2.1
    Python 3.6.2

Sources :

1) Beckham, C., & Pal, C. (2016). A simple squared-error reformulation for
  ordinal classification. arXiv preprint arXiv:1612.00775.

2) Dosovitskiy, A., Fischer, P., Ilg, E., Hausser, P., Hazirbas, C., Golkov, V.,
  ... & Brox, T. (2015). Flownet: Learning optical flow with convolutional networks.
  In Proceedings of the IEEE International Conference on Computer Vision (pp. 2758-2766).

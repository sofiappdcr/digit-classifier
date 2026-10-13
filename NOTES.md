#Project Notes:

## Sprint 1: loading and exploring the data.

**What was done:** downloaded MNIST and displayed the first 10 images with their labels.

**What was learned:**
-Each image is a 28x28 grid of pixels, flattened into a row of 784 numbers (from 0 to 255 - 0: black, 255: white, in-between: shades of them).
-X holds the image, y holds the correct answer (digit) for each one.
-reshape turns the row back into a square so it can be drawn (change initially made my MNIST).

**Problems I ran into:** libraries installed in the wrong version of Python. The problem was fixed by using a virtual environment recommended by VS Code.

**Next Step:** train a first model and measure its accuracy.

## Sprint 2: training the 1st model.
**What was done:** 
1. load_digits: first data set used, 8x8 images and 1,797 of them, but only 1,437 (80%) used for training, while the other 20% (360) was used for testing.

->logistic regression: accuracy of 97.22% (test) (99.93%-training)and 0.0s of training.
(mean of 5 tests: 0.9706 ± 0.0065).
->neural network: accuracy of 98,06% (test) (100%-training) and 0.2s of training.
(mean of 5 tests: 0.9750 ± 0.0030).
->conclusion: tie, with a slight advantage for the neural network.

2. MNIST: second data set used, 28x28 images and 70,000 of them, but only 56,000 (80%) used for training, while the other 20% (14.000) was used for testing.

->logistic regression: accuracy of 91.66% (test) (94.48%-training) and 3.4s of training.
(mean of 5 tests: 0.9168 ± 0.0020).
->neural network: accuracy of 97.06% (test) (100%-training) and 6.5s of training.
(mean of 5 tests: 0.9705 ± 0.0021).
->conclusion: noticeably, the neural network presented better results when comparing to the logistic regression.

**What was learned:**
-Logistic Regression: it is a neural network with no hidden layers (ex of a hidden layer: one with 64 neurons that identifies curves and lines, in this case). it is fast to train, because it has few parameters. it is a linear function.
-ReLU (Rectified Linear Unit): it's a non-linear function used by neural networks.
-The hidden layer and the non-linearity presented by ReLU is what gives the neural network the hability to learn patterns that logistic regression can't.
-Expectation: neural network has better accuracy than logistic regression (better seen with the MNIST data set).


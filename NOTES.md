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

## Sprint 3: training curve plotting, confusion matrix and .
**What was done:** 
1. TASK 1: the network was trained one epoch at a time and each metrics were recorded after each epoch and plotted in a chart.
->(8x8, 1,797 samples) | Train: 1077 | Validation: 360 | Test: 360
    Baseline (64) -> val accuracy = 97.5%, time=0.2s. # baseline accuracy is higher than final validation, because baseline stops only at epoch 177 (the loss stopped improving), whilst the other stops at 100 epochs. the gap is 3 images out of 360, which is within noise.
    Final train acc = 99.91%.
    Final val acc = 96.67 %.
    Best val acc = 96.67% (epoch 76).
->chart interpretation:
    -loss dropped fast then flattened out.
    -both training and validation were high and close together, meaning the training worked well, with slight overfitting.
2. TASK 2: the confusion matrix was made using the baseline model (177 epoches).
->9 errors/360 (97.5% from baseline).
predicted/real: 1/9 (twice); 5/6 (once); 4/8 (once); 7/2 (once); 5/8 (once); 0/4 (once); 1/6 (once); 1/2 (once).
3. TASK 3: experiments.md was created with 10 experiments, each one changing one thing from the baseline.
-> _lower learning rate_ was clearly worse (95%).
-> _more neurons (256)_ helped a little. it was the best model (98.43% on val) by comparison, but only 0.6 points above the baseline.
-> _too few neurons (32)_ hurt by 1 point.
-> _two layers_ did not help. the problem is probably simple enough that one layer is plenty.
-> _regularization_ had almost no effect.
-> _every row reaches 100% train accuracy_ (except the slow one), so all of them overfit a little.
->**Final Result**: the model chosen by validation was the one with 256 neurons. on the test set it got 97.78% (352/360, only 8 wrong).

**What was learned:**
-Each full pass over the training data is called an _epoch_.
-_loss_: measures how wrong the network is, even when it is right but not confident. it is what the network tries to minimize during training.
-_accuracy_: it is the % of correct predictions.
-train vs. validation on the chart:
1. close together and high-the model learned and generalizes well.
2. train high and validation well below-overfitting, meaning the model is memorizing.
3. both low-the model is too simple or was trained too little.
-_regularization_ is a defense against overfitting.
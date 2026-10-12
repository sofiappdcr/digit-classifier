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

**What was learned:**
-Logistic Regression: it is a neural network with no hidden layers (ex: one with 64 neurons that identifies curves and lines, in this case). it is a sigmoid function with a much higher computational cost if you compare it with ReLU.
-ReLU (Rectified Linear Unit): it's a function that gives the neural networks the hability to learn patterns that logistic regression can't.
-Expectation: neural network has better accuracy than logistic regression.

#Project Notes:

## Sprint 1: loading and exploring the data.

**What was done:** downloaded MNIST and displayed the first 10 images with their labels.

**What was learned:**
-Each image is a 28x28 grid of pixels, flattened into a row of 784 numbers (from 0 to 255 - 0: black, 255: white, in-between: shades of them).
-X holds the image, y holds the correct answer (digit) for each one.
-reshape turns the row back into a square so it can be drawn (change initially made my MNIST).

**Problems I ran into:** libraries installed in the wrong version of Python. The problem was fixed by using a virtual environment recommended by VS Code.

**Next Step:** train a first model and measure its accuracy.
# Automatic differentiation

Previously we have calculated our backpropagation by hand.
We rearrange our Day 2 example to benefit from _automatic differentiation_.
[The corresponding tutorial](http://penkovsky.com/post/neural-networks-3/) about neural networks.

Finally, we treat both linear layers and activations as layers.
That allows us to introduce new kinds of operations (batch normalization, dropout)
That point of view will be useful in the future when considering
convolutional neural networks (conv, pooling layers).


## How To Build

1. Install stack:

     ```
     $ wget -qO- https://get.haskellstack.org/ | sh
     ```

(alternatively, `curl -sSL https://get.haskellstack.org/ | sh`)

2. Install open-blas from https://www.openblas.net/ (needed for hmatrix package)

3. Compile and run

     ```
     $ ./run.sh
     ```

```
Circles problem, 1 hidden layer of 128 neurons, 1000 epochs
---
Training accuracy (gradient descent) 64.5
Validation accuracy (gradient descent) 59.0

Training accuracy (Adam) 99.0
Validation accuracy (Adam) 99.0


Spirals problem, Adam, 700 epochs
---
1 hidden layer, 128 neurons (513 parameters)
Training accuracy 84.2
Validation accuracy 83.0

1 hidden layer, 512 neurons (2049 parameters)
Training accuracy 87.8
Validation accuracy 85.0

3 hidden layers, 40, 25, and 10 neurons (1416 parameters)
Training accuracy 100.0
Validation accuracy 100.0
```

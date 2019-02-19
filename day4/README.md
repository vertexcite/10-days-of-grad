# Automatic differentiation

[The corresponding tutorial](http://penkovsky.com/post/neural-networks-4/) about neural networks.

Our roadmap:
new operations (batch normalization, dropout);
convolutional neural networks (conv, pooling layers).
Today, we will consider batch normalization.


## How To Build

0. Download MNIST data into data/ directory.

1. Install stack:

     ```
     $ wget -qO- https://get.haskellstack.org/ | sh
     ```

(alternatively, `curl -sSL https://get.haskellstack.org/ | sh`)

2. Install open-blas from https://www.openblas.net/ (needed for hmatrix package)

3. Compile and run

     ```
     stack build
     stack exec mnist
     ```

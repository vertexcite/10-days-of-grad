# Batch normalization

## How To Build

1. Install stack:

     ```
     $ wget -qO- https://get.haskellstack.org/ | sh
     ```

(alternatively, `curl -sSL https://get.haskellstack.org/ | sh`)

2. Build and run MNIST benchmark on all available cores:

     ```
     $ ./run.sh
     ```

     10 training epochs

     Training accuracy (SGD + batchnorm) 99.0
     Validation accuracy (SGD + batchnorm) 97.9

     Training accuracy (SGD) 47.6
     Validation accuracy (SGD) 47.0

Note that in the last case (SGD), the weights are initialized the same way as
in the first one (SGD + batchnorm). This initialization is not optimal.

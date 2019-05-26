# Batch normalization

## How To Build

1. Install stack:

     ```
     $ wget -qO- https://get.haskellstack.org/ | sh
     ```

(alternatively, `curl -sSL https://get.haskellstack.org/ | sh`)

2. Build and run

     ```
     $ stack build
     $ stack exec main-exe
     ```

Results of 10 runs with different weight initialization each time:


     ```
     $ for i in `seq 1 10`; time stack exec main-exe -- +RTS -N2

     Circles problem, 1 hidden layer of 128 neurons, 200 epochs
     ---
     Training accuracy (SGD + batchnorm) 84.9
     Validation accuracy (SGD + batchnorm) 86.5
     Training accuracy (SGD) 76.0
     Validation accuracy (SGD) 74.0
     stack exec main-exe -- +RTS -N2  4.91s user 0.91s system 168% cpu 3.443 total

     Circles problem, 1 hidden layer of 128 neurons, 200 epochs
     ---
     Training accuracy (SGD + batchnorm) 100.0
     Validation accuracy (SGD + batchnorm) 100.0
     Training accuracy (SGD) 69.3
     Validation accuracy (SGD) 74.0
     stack exec main-exe -- +RTS -N2  5.17s user 1.10s system 168% cpu 3.721 total

     Circles problem, 1 hidden layer of 128 neurons, 200 epochs
     ---
     Training accuracy (SGD + batchnorm) 99.5
     Validation accuracy (SGD + batchnorm) 100.0
     Training accuracy (SGD) 67.7
     Validation accuracy (SGD) 67.7
     stack exec main-exe -- +RTS -N2  5.15s user 1.05s system 167% cpu 3.699 total

     Circles problem, 1 hidden layer of 128 neurons, 200 epochs
     ---
     Training accuracy (SGD + batchnorm) 100.0
     Validation accuracy (SGD + batchnorm) 100.0
     Training accuracy (SGD) 82.3
     Validation accuracy (SGD) 72.9
     stack exec main-exe -- +RTS -N2  5.04s user 1.04s system 168% cpu 3.615 total

     Circles problem, 1 hidden layer of 128 neurons, 200 epochs
     ---
     Training accuracy (SGD + batchnorm) 100.0
     Validation accuracy (SGD + batchnorm) 100.0
     Training accuracy (SGD) 92.2
     Validation accuracy (SGD) 89.6
     stack exec main-exe -- +RTS -N2  5.23s user 0.97s system 169% cpu 3.668 total

     Circles problem, 1 hidden layer of 128 neurons, 200 epochs
     ---
     Training accuracy (SGD + batchnorm) 83.9
     Validation accuracy (SGD + batchnorm) 75.0
     Training accuracy (SGD) 93.2
     Validation accuracy (SGD) 95.8
     stack exec main-exe -- +RTS -N2  5.21s user 1.03s system 168% cpu 3.706 total

     Circles problem, 1 hidden layer of 128 neurons, 200 epochs
     ---
     Training accuracy (SGD + batchnorm) 84.9
     Validation accuracy (SGD + batchnorm) 78.1
     Training accuracy (SGD) 77.1
     Validation accuracy (SGD) 79.2
     stack exec main-exe -- +RTS -N2  5.08s user 0.95s system 167% cpu 3.597 total

     Circles problem, 1 hidden layer of 128 neurons, 200 epochs
     ---
     Training accuracy (SGD + batchnorm) 100.0
     Validation accuracy (SGD + batchnorm) 99.0
     Training accuracy (SGD) 80.7
     Validation accuracy (SGD) 76.0
     stack exec main-exe -- +RTS -N2  5.02s user 0.97s system 168% cpu 3.548 total

     Circles problem, 1 hidden layer of 128 neurons, 200 epochs
     ---
     Training accuracy (SGD + batchnorm) 99.5
     Validation accuracy (SGD + batchnorm) 99.0
     Training accuracy (SGD) 92.7
     Validation accuracy (SGD) 88.5
     stack exec main-exe -- +RTS -N2  5.08s user 1.04s system 167% cpu 3.658 total

     Circles problem, 1 hidden layer of 128 neurons, 200 epochs
     ---
     Training accuracy (SGD + batchnorm) 92.7
     Validation accuracy (SGD + batchnorm) 90.6
     Training accuracy (SGD) 90.1
     Validation accuracy (SGD) 92.7
     stack exec main-exe -- +RTS -N2  5.29s user 1.07s system 166% cpu 3.814 total
     ```

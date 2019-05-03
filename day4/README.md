# Haskell Neural Network on GPU

[The tutorial](http://penkovsky.com/post/neural-networks-2/) about neural networks.

## How To Build

1. Install stack:

     ```
     $ wget -qO- https://get.haskellstack.org/ | sh
     ```

(alternatively, `curl -sSL https://get.haskellstack.org/ | sh`)

2. Install CUDA
3. Install LLVM

    Ubuntu 18.04:

    $ sudo apt install llvm-6.0-dev

    Arch Linux:

    $ sudo pacman -S llvm7

4. Export LLVM/CUDA paths

    $ export PATH=/usr/lib/llvm-6.0/bin:$PATH
    $ export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
    $ export LD_LIBRARY_PATH=/usr/local/cuda/nvvm/lib64:$LD_LIBRARY_PATH

5. Compile and run

     ```
     $ ./run.sh
     ```

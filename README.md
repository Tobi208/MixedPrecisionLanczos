# Mixed-Precision Lanczos on HPC Architecture

Basic and Block Lanczos in mixed-precision on NVIDIA HPC architectures. Tested on CUDA Toolkit version 12.3.0. Master Thesis in Computer Science at Uppsala University, 2024.

## Generate Test Data as Symmetric COO

```
make gen
./gen_sym_coo <size> <density> <min val> <max val> <precision[double|single|half]> <output>
```

## Run Algorithms

```
make lanczos
./lanczos <file [symmetric coo | csr]> basic <precision [D|S|Mmin|Mopt]> <iterations>
./lanczos <file [symmetric coo | csr]> block <precision [D|S|Mmin|Mopt]> <iterations> <blocksize>
```

## Reproducability

`lanczos.cpp` contains readable code meant to help the reader understand the project whereas `experiments/` contains the code used for the experiments but the macros need to be configured manually and the files compiled separately.

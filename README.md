# Driven1dRydberg


Driven1dRydberg contains the codes for reproducing the results contained in https://arxiv.org/abs/2309.12392. 

The codes are written both in C++, with taylored functions writte on top of the ITensor v3 library for tensor network methods, and Python, based on the quimb Python package.


## Prerequisites

C++ ITensor v3 library: You need the ITensor v3 library to be installed. See the ITensor Installation Guide for details https://itensor.org/docs.cgi?vers=cppv3&page=install

C++17: This library requires a C++17-compliant compiler.

quimb: needed the Python package quimb for Exact Diagonalization and Tensor Network methods (written in Python). Install it via the command
```bash
pip install quimb
```

## Installation
Clone the repository:
```bash
git clone https://github.com/rvalencia1995/Driven1dRydberg.git
```

## Usage
Modify path to your ITensor library in order to compile via 'make' command the .cpp codes.

## License

[MIT](https://choosealicense.com/licenses/mit/)

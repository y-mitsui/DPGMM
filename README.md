DPGMM
=============

C liblary of Variational Inference for the Infinite Gaussian Mixture Model of haines/DPGMM base.

## Features
* Improve parformance


## Install
You require cmake and GSL

**[Case of Linux]**  
1. Input this following on shell.  

    $ cmake .     #(Be careful dot.)
    $ make && make install

**[Case of MinGW on Windows]**  
1. Input this following on command prompt.

    $ PATH=C:\MinGW\bin;%PATH%
    $ cmake -G "MinGW Makefiles"
    (input one more)
    $ cmake -G "MinGW Makefiles"
    $  mingw32-make

2. Add "libdpgmm.a" and "dpgmm.h"  to your project.

## Usage
Read example.c

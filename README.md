DPGMM
=============

C liblary of Variational Inference for the Infinite Gaussian Mixture Model of haines/DPGMM base.

## Features
* Improved performance, because of written by C language only.


## Install
You require cmake and GSL

**[Case of Linux]**  
1. Input this following on shell.  

    $ cmake .     #(Be careful dot.)
    $ make && sudo make install

**[Case of MinGW on Windows]**  
1. Input this following on command prompt.

    $ PATH=C:\MinGW\bin;%PATH%
    $ cmake -G "MinGW Makefiles"
    (input one more)
    $ cmake -G "MinGW Makefiles"
    $  mingw32-make

2. Add "libdpgmm.a" and "dpgmm.h"  to your project.

## Usage
Read dpgmm.h and example.c

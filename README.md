﻿DPGMM
=============

C library of Variational Inference for the Infinite Gaussian Mixture Model of haines/DPGMM base.

## Features
* Improved performance, because written by C language only.


## Install
You require cmake and GSL

**[Case of Ubuntu]**  
(1) Install GNU Scientific Library (GSL) using apt-get 

    $ sudo apt-get install libgsl0ldbl libgsl0-dev

(2) Input this following on shell.

    $ cmake .     
    $ make && sudo make install

(3) Add so searching path /usr/local/lib

    $ sudo vim /etc/ld.so.conf
    $ ldconfig

## Usage
This library supported pkg-config.

    $ gcc `pkg-config --cflags dpgmm` yourProgram.c `pkg-config --libs dpgmm`

Read dpgmm.h and example2.c

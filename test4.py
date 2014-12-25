#! /usr/bin/env python
import os
import sys
from gcp.gaussian import Gaussian
from dpgmm import DPGMM
import numpy.random
import time
from scipy.integrate import quad
import csv

dims=2
model = DPGMM(dims, 7)
spamReader = csv.reader(open('data4.txt', 'rb'), delimiter=' ', quotechar='|')
for row in spamReader:
  model.add([float(row[0]),float(row[1])])
model.setPrior()
model.solve()
sride=200
rangeX=[-1.0,18.0]
rangeY=[0.98,1.03]
for i in range(sride):
  x=float(i)/float(sride)*(rangeX[1]-rangeX[0])+rangeX[0]
  for j in range(sride):
    y=float(j)/float(sride)*(rangeY[1]-rangeY[0])+rangeY[0]
#    print "%f %f %f"%(x,y,model.prob([x,y]))


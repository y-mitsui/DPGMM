#! /usr/bin/env python
import os
import sys
from gcp.gaussian import Gaussian
from dpgmm import DPGMM
import numpy.random

dims=2
numpy.random.seed(1);
gt = Gaussian(dims)
gt.setMean([1.0,0.0])
gt.setCovariance([[1.0,0.8],[0.8,1.0]])
sample_count = 10
sample=[]
for _ in xrange(sample_count):
  sample.append(gt.sample())

#f=open('data.txt','w')
#for x in sample:
#  f.write('%lf,%lf\n'%(x[0],x[1]))
#f.close()
model = DPGMM(dims, 1)
for i,data in enumerate(sample):
  model.add(data)
model.setPrior()
#model.solve()
#for i in range(10):
#  x=i*0.4-2.0
  #print "%f,%f"%(x,model.prob([x]))


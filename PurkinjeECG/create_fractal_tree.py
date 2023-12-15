# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:02:34 2015

@author: fsc
"""

from FractalTree import *
from parameters import Parameters
from time import time
param=Parameters()
param.save = False
param.meshfile = "data/sphere.obj"

tstart = time()
branches, nodes = Fractal_Tree_3D(param)
print(f"Time: {time()-tstart}")


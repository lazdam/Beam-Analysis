#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import healpy as hp
from astropy import units
from scipy.interpolate import interp1d, pchip, interp2d
import ephem
from datetime import datetime
from pkg_resources import resource_filename
from pygsm import GlobalSkyModel
from pygsm2016 import GlobalSkyModel2016
import matplotlib.pyplot as plt
import scipy.stats
from pylab import arange, show, cm
import h5py as h5


#Create gsm object. 
gsm = GlobalSkyModel()

#Generate the maps. Going from 50MHz-100MHz with increments of 1MHz
gsm_map = gsm.generate(np.linspace(50,100,51))

#Apply Singular Value Decomposition
U,S,V = np.linalg.svd(gsm_map, 0)

#Plot first 4 principle components
for i in range(0,4):
    if i < 3:
        plt.plot(U[:,i], label = 'Mode '+str(i))
    else: 
        plt.plot(U[:,i], label = 'Mode '+str(i), alpha = 0.5)
plt.legend()
plt.show()


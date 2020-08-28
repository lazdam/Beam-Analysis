#!/usr/bin/env python
# coding: utf-8

# In[11]:


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

#First, create a GlobalSkyModel object
gsm = GlobalSkyModel()

#Now, use the .generate method to generate the map at a specified frequency. The default coordinate system is Galactic, default frequency unit is MHz. 
ν = 70 #MHz
map_at_70MHz = gsm.generate(ν)

#To view the map you generated, use healpy.mollview(map). Note that 'cmap' just changes the colors of the plot.
hp.mollview(map_at_70MHz, cmap = 'coolwarm')

#You should notice that the map is relatively dark. To make it more visible, take the log2 of the map to bring out detail
hp.mollview(np.log2(map_at_70MHz), cmap = 'coolwarm')



#Now, let's do the same thing but for gsm 2016

#This time, we have a few more options. We will specify the resolution and the temperature units. I tend to use high resolution and TCMB units. 
gsm_2016 = GlobalSkyModel2016(resolution = 'hi', unit = 'TCMB')

#From here, everything else is the same
ν = 70 #Mhz
map_2016_at_70MHz = gsm_2016.generate(ν)
hp.mollview(map_2016_at_70MHz, cmap = 'coolwarm')
hp.mollview(np.log2(map_2016_at_70MHz), cmap = 'coolwarm')


# In[ ]:





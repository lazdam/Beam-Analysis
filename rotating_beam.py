#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import healpy
import matplotlib.pyplot as plt

#Generate beam same way as before
nside = 2**8
npix = healpy.nside2npix(nside)
pix_array = np.arange(npix)
θ, ϕ = healpy.pix2ang(nside, pix_array)
xyz = np.array([np.cos(ϕ)*np.sin(θ), np.sin(ϕ)*np.sin(θ), np.cos(θ)])
beam_cent = (np.radians(90), np.radians(0)) 
beam_cent_xyz = healpy.dir2vec(beam_cent)
beam_x_vals = np.dot(beam_cent_xyz, xyz)

fwhm = 40
sig = fwhm/np.sqrt(8*np.log(2))*np.pi/180
beam = np.exp(-0.5*(np.arccos(beam_x_vals))**2/sig**2)

#Up until here, we haven't changed anything and the beam should appear the same as before. 
#We will now rotate the beam. 

#First, convert the beam to alm space
alm_beam = healpy.map2alm(beam)

#Determine l_max
lmax = np.int(np.round( np.sqrt(2*len(alm_beam))-0.5))

#Determine mvec. 
#Note about Healpix: When you convert from map space to alm space, it arranges the alm coefficients in such a way 
#that the first lmax + 1 coefficients correspond to m = 0, the next lmax coefficients to m = 1, next lmax - 1 to 
# m = 2 etc... until you reach the last 1 coefficient for m = lmax. 


mvec = np.zeros(len(alm_beam))

icur = 0

for i in range(lmax):
    nm = lmax - i 
    mvec[icur:icur+nm] = i
    icur+=nm

#Apply rotation
ϕ = np.pi/2
rotated_alms = alm_beam*np.exp(1j*mvec*ϕ) 

#Convert back to map space
rotated_beam = healpy.alm2map(rotated_alms, nside)

healpy.mollview(rotated_beam)


# In[ ]:





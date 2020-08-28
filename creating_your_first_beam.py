#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import healpy

#First, set your nside (feel free to change this and notice the difference between nside = 4 and nside = 512)
nside = 2**8

#Determine the number of pixels
npix = healpy.nside2npix(nside)

#Create an array of size npix to represent each pixel
pix_array = np.arange(npix)

#Determine the angular position of each pixel
θ, ϕ = healpy.pix2ang(nside, pix_array)

#Use θ, ϕ to find the cartesian coordinate using spherical change of coordinates
xyz = np.array([np.cos(ϕ)*np.sin(θ), np.sin(ϕ)*np.sin(θ), np.cos(θ)])

#Set where you'd like to center your beam. Change this to become familiar with how the angles work. 
beam_cent = (np.radians(90), np.radians(0)) #(θ, ϕ). True North represents θ = 0. 

#Note: From now on, we will not change the value of phi using beam_cent and it will always be set to 0.

#Convert the center of the beam to cartesian
beam_cent_xyz = healpy.dir2vec(beam_cent)

#Determine the x-values of the beam by taking the dot product between the beam center and the pixels
beam_x_vals = np.dot(beam_cent_xyz, xyz)

#Calculate the beam
fwhm = 40
sig = fwhm/np.sqrt(8*np.log(2))*np.pi/180
beam = np.exp(-0.5*(np.arccos(beam_x_vals))**2/sig**2)

#Mollweide view
healpy.mollview(beam)


# In[ ]:





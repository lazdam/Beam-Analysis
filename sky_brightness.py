#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# A note before we start: From here on out, we will be working in equatorial coordinates. By default, all of the maps from pygsm and pygsm2016 are in galactic coordinates, so before we can use them, you have to change them. Changing the map is quite easy and can be done using the Rotator class from healpy. Here is a simple example on how it is done: 

# In[6]:


gsm = GlobalSkyModel()
ν = 70 #mhz
map_galactic = gsm.generate(ν)

#Create a rotate_map object of class Rotator
rotate_map = hp.rotator.Rotator(coord = ['G', 'C']) #G for Galactic, C for Equatorial

#Use the .rotate_map_pixel method to rotate the map
map_equatorial = rotate_map.rotate_map_pixel(map_galactic)

hp.mollview(map_equatorial)


# That's all there is to it. Be careful not to think of 'E' for equatorial, as 'E' is for ecliptic coordinates. 

# ## Method 1: Working in Map Space

# In[ ]:


#Step 1: Generate a map of the sky and convert it to equatorial

gsm = GlobalSkyModel()
ν = 70
gsm_map_gal = gsm.generate(ν)
rotate_map = hp.rotate.Rotate(coord = ['G', 'C'])
gsm_map_eq = rotate_map.rotate_map_pixel(gsm_map_gal)

#Step 2: Set up the foundations of the beam
nside = 2**9
th, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
xyz = np.zeros([hp.nside2npix(nside),3])
xyz[:,0] = np.sin(th)*np.cos(phi)
xyz[:,1] = np.sin(th)*np.sin(phi)
xyz[:,2] = np.cos(th)

#Step 3: Determine your phi rotations. The more you do, the more points you will have, the smoother the graph
ϕ_rotations = np.linspace(0, 2*np.pi, 100)

#Create an empty list for your to-be-calculated temperatures
temperatures = []

#Calculate the temperature for each rotation of ϕ

for ϕ in ϕ_rotations: 
    
    beam_cent = np.radians(137), -ϕ #theta, phi
    beam_cent_vec = hp.dir2vec(beam_cent)

    sky_area = np.dot(xyz, beam_cent_vec) #x-values of beam
    beam_width = 0.35

    beam = np.exp(-0.5*(sky_area-1)**2/beam_width**2)
    
    temp_map_space = np.sum(BEAM*gsm_map_eq)/np.sum(BEAM) #From the equation in the report
    
    temperatures.append(temp_map_space)
    
#Plot your results
plt.plot(ϕ_rotations,temperatures)
plt.title(f'Amplitude as a Function of Time in Map Space for Sky at {ν}MHz')
plt.xlabel('Time, ϕ (rad)')
plt.ylabel('Amplitude, T (K)')
plt.show()


# # Method 2: Using Fast Fourier Transform

# In[13]:


#Beginning is all the same
gsm = GlobalSkyModel()
ν = 70
gsm_map_gal = gsm.generate(ν)
rotate_map = hp.rotator.Rotator(coord = ['G', 'C'])
gsm_map_eq = rotate_map.rotate_map_pixel(gsm_map_gal)

#Convert to cartesian
nside = 2**9
th, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
xyz = np.zeros([hp.nside2npix(nside),3])
xyz[:,0] = np.sin(th)*np.cos(phi)
xyz[:,1] = np.sin(th)*np.sin(phi)
xyz[:,2] = np.cos(th)

#Set center of beam
beam_cent = np.radians(137), np.radians(0) #theta, phi
beam_cent_vec = hp.dir2vec(beam_cent)

sky_area = np.dot(xyz, beam_cent_vec) #x-values of beam


#Gaussian beam
beam_width = 0.35
beam = np.exp(-0.5*(sky_area-1)**2/beam_width**2)

#This is where it starts to differ
#Convert to ALM space, both the map and the beam.
alm_beam = hp.map2alm(beam)
alm_map = hp.map2alm(gsm_map_eq)

#Here we sum the alms over m, and multiply each value by the conjugate of the other.
def collapse_alm_prod(alms1,alms2, padded = 0):
    lmax=nalm2lmax(len(alms1))
    vec=np.zeros(lmax+1,dtype='complex')
    icur=0
    
    for m in range(lmax):
        nm=lmax-m
        vec[m]=np.sum((alms1[icur:icur+nm])*np.conj(alms2[icur:icur+nm]))
        icur=icur+nm
        
    y = np.zeros(padded)
    z = np.hstack([vec,y])
    return z

#Computes the lmax of a set of alms
def nalm2lmax(nalm):
    lmax=np.int(np.round( np.sqrt(2*nalm)-0.5))
    return lmax

#Computes the mvec discussed in the previous section
def get_mvec(alms):
    #lmax=np.int(np.round( np.sqrt(2*len(alms))-0.5))
    lmax=nalm2lmax(len(alms))
    mvec=np.zeros(len(alms))
    icur=0
    for m in range(lmax):
        nm=lmax-m
        mvec[icur:icur+nm]=m
        icur=icur+nm
    #print(icur,len(alms))
    return mvec

z_m = collapse_alm_prod(alm_beam, alm_map)


#Find iFFT, which gives us the average brightness as a function of time over one day
def beam_integral_fullday_from_collapsed(vec,alm0):
    tmp=vec.copy()
    tmp[1:]=2*tmp[1:]
    vecft=np.fft.ifft(tmp)
    mynorm=np.sqrt(4*np.pi)*alm0
    ans=(vecft/mynorm)
    ans=ans*len(ans) #need this normalization factor to undo the one built into the ifft
    phi = np.linspace(0,2*np.pi,len(ans))
    return ans,phi

zmft,phi = beam_integral_fullday_from_collapsed(z_m,alm_beam[0])

plt.plot(phi, np.real(zmft), color = 'purple')
plt.xlabel('Time, ϕ (rad)')
plt.ylabel('Amplitude, T (K)')
plt.savefig('temp 1 day')
plt.show()


# In[ ]:





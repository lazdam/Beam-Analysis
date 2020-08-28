#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from matplotlib import pyplot as plt

import healpy
import pygsm
import scipy
from pygsm import GSMObserver
from pygsm import GlobalSkyModel
from pygsm2016 import GlobalSkyModel2016
plt.ion()

def nalm2lmax(nalm):
    lmax=np.int(np.round( np.sqrt(2*nalm)-0.5))
    return lmax


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

def rotate_beam_phi(beam,phi):
    nside=healpy.npix2nside(len(beam))
    alm_beam=healpy.map2alm(beam)
    mvec=get_mvec(alm_beam)
    alm_rot=alm_beam*np.exp(1J*mvec*phi)
    beam_rot=healpy.alm2map(alm_rot,nside)
    return beam_rot 


def collapse_alm_prod(alms1,alms2):
    lmax=nalm2lmax(len(alms1))
    vec=np.zeros(lmax+1,dtype='complex')
    icur=0
    for m in range(lmax):
        nm=lmax-m
        vec[m]=np.sum((alms1[icur:icur+nm])*np.conj(alms2[icur:icur+nm]))
        icur=icur+nm
    return vec

def beam_integral_from_collapsed(vec,alm0,phi):
    mvec=np.arange(len(vec))
    vec_rot=vec*np.exp(1J*mvec*phi)
    normfac=np.sqrt(4*np.pi)*alm0
    return np.real( (vec_rot[0]+2*np.sum(vec_rot[1:]))/normfac)

def beam_integral_fullday_from_collapsed(vec,alm0):
    tmp=vec.copy()
    tmp[1:]=2*tmp[1:]
    vecft=np.fft.ifft(tmp)
    mynorm=np.sqrt(4*np.pi)*alm0
    ans=np.real(vecft/mynorm)
    ans=ans*len(ans) #need this normalization factor to undo the one built into the ifft
    phi=2*np.pi*np.arange(len(ans))/len(ans)
    return ans,phi


# # Part 1: Find the Center of Your Beam

# In[ ]:


nside = 2**10
th, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
xyz = np.array([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)])

#I initiall started with rotations from 0 - 2π: however, I narrowed it down
#to between these two values. 
rots = np.linspace(0.63, 0.70, 100)

#Get baseline
ant1 = [-46.88687, 37.82053 ]
ant2 = [-46.886733,37.81902 ]

re=40e6/2/np.pi

#Converts lat/lon to xyz
ant1_xyz=np.asarray([np.cos(ant1[0]*np.pi/180)*np.cos(ant1[1]*np.pi/180),np.cos(ant1[0]*np.pi/180)*np.sin(ant1[1]*np.pi/180),np.sin(ant1[0]*np.pi/180)])
ant2_xyz=np.asarray([np.cos(ant2[0]*np.pi/180)*np.cos(ant2[1]*np.pi/180),np.cos(ant2[0]*np.pi/180)*np.sin(ant2[1]*np.pi/180),np.sin(ant2[0]*np.pi/180)])

#Baseline
bl=(ant1_xyz-ant2_xyz)*re

u_th = np.dot(bl, xyz)

angles = []

for i in rots: 
    cent = (np.radians(137), i)
    vec = hp.dir2vec(cent)
    baseline = np.dot(vec, xyz)

    #Beam
    sigma = 0.00001
    beam=np.exp(-0.5*(np.arccos(baseline))**2/sigma**2)
    
    x = np.dot(u_th, beam)
    
    angles.append(x)

#On the graph, find the rotation that causes the y value to be 0. 
#That will be how much you have to rotate your beam from the center. 
plt.plot(rots, angles)
plt.show()


# # Part 2: Amplitude and Phase Plots

# First, you're going to need the position of two antennas. In the case of PRIzM, we have the lat/lon. We're going to convert this to cartesian and use this to determine the baseline. For theory, see Prof. Sievers. 

# In[7]:


#Lat and lon of both antennas
nside = 2**9
ant1=[-46.88687, 37.82053 ]
ant2=[-46.886733,037.81902 ]

#Convert to xyz
re=40e6/2/np.pi
ant1_xyz=np.asarray([np.cos(ant1[0]*np.pi/180)*np.cos(ant1[1]*np.pi/180),np.cos(ant1[0]*np.pi/180)*np.sin(ant1[1]*np.pi/180),np.sin(ant1[0]*np.pi/180)])
ant2_xyz=np.asarray([np.cos(ant2[0]*np.pi/180)*np.cos(ant2[1]*np.pi/180),np.cos(ant2[0]*np.pi/180)*np.sin(ant2[1]*np.pi/180),np.sin(ant2[0]*np.pi/180)])

#Find the baseline
bl=(ant1_xyz-ant2_xyz)*re


# Now that we have the baseline, we are going to calculate the phase. The phase is dependent on the baseline as well as frequency. Feel free to change the frequency as you see fit and see how it changes the mollview. 

# In[9]:


#Get the map ready
th,phi=healpy.pix2ang(nside,np.arange(healpy.nside2npix(nside)))
xyz=np.empty([len(th),3])
xyz[:,0]=np.sin(th)*np.cos(phi)
xyz[:,1]=np.sin(th)*np.sin(phi)
xyz[:,2]=np.cos(th)

ν = 30 #MHz
λ = 300/ν

#Find the phase
phase = np.exp(1j*2*np.pi*np.dot(xyz, bl)/λ)

hp.mollview(np.real(phase))


# The last thing we'll do is introduce a beam that will cover a section of the plot above. I'll use a gaussian beam here centered at the center I found in part 1. 

# In[12]:


beam_cent = healpy.dir2vec(np.radians(137), 0.65969)
sigma = 0.4
beam=np.exp(-0.5*(np.arccos(np.dot(xyz, beam_cent))**2/sigma**2))

#Multiply the beam and the phase together
hp.mollview(beam*phase)

#Notice that the beam covers only the area where the lines are relatively parallel to one another


# In[2]:


def phase_plot(nside, freqs, path_to_beam, path_to_gsm, pol = 'NS' ):
    
    '''Function takes in nside, frequency and polarization and outputs 
    matrices containing information for the amplitude and phase plots. 
    
    Parameters
    ------------
    1. nside (int): Must match the nside of maps you are using.
    2. freqs (array): numpy array of frequencies.
    3. path_to_beam (string): Path to folder containing the simulated beams.
    4. path_to_gsm (string): Path to folder containing gsm map alms. 
    5. pol (string): NS for North South polarization, EW for East-West polarization. This function assumes you've
       saved your simulated beams to your computer.'''
    
    #Lat and lon of both antennas
    ant1=[-46.88687, 37.82053 ]
    ant2=[-46.886733,037.81902 ]

    #Convert to xyz
    re=40e6/2/np.pi
    ant1_xyz=np.asarray([np.cos(ant1[0]*np.pi/180)*np.cos(ant1[1]*np.pi/180),np.cos(ant1[0]*np.pi/180)*np.sin(ant1[1]*np.pi/180),np.sin(ant1[0]*np.pi/180)])
    ant2_xyz=np.asarray([np.cos(ant2[0]*np.pi/180)*np.cos(ant2[1]*np.pi/180),np.cos(ant2[0]*np.pi/180)*np.sin(ant2[1]*np.pi/180),np.sin(ant2[0]*np.pi/180)])

    bl=(ant1_xyz-ant2_xyz)*re

    th,phi=healpy.pix2ang(nside,np.arange(healpy.nside2npix(nside)))
    xyz=np.empty([len(th),3])
    xyz[:,0]=np.sin(th)*np.cos(phi)
    xyz[:,1]=np.sin(th)*np.sin(phi)
    xyz[:,2]=np.cos(th)


    #TEST BLOCK 
    #--------------------------------------------------------#
    
    #Load in beam. CHANGE THIS IF USING DIFFERENT BEAM
    beam = np.load(path_to_beam + f'beam_30.0mhz_{pol}.npy')
    
    #Load in map alms. Change name of file to whatever you named your files
    alms_gsm = np.load(path_to_gsm + f'gsm_map_{30.0}mhz_2016_alm.npy')
    alms_beam = healpy.map2alm(beam)
    
    #Set the frequency
    nu=30
    lamda=300/nu
    
    #Set your phase
    myphase=np.exp(2*np.pi*1J*np.dot(xyz,bl)/lamda)
    
    #Solve real and imaginary parts separately
    
    #Real: 
    alms_real=healpy.map2alm(np.real(myphase)*beam)
    vec_real=collapse_alm_prod(alms_real,alms_gsm)
    vis_real,phi_real=beam_integral_fullday_from_collapsed(vec_real,alms_beam[0])

    #Imag: 
    alms_im=healpy.map2alm(np.imag(myphase)*beam)
    vec_im=collapse_alm_prod(alms_im,alms_gsm)
    vis_im,phi_im=beam_integral_fullday_from_collapsed(vec_im,alms_beam[0])
    
    #---------------------------------------------------------#
    
    

    nu=freqs
    
    nu_base = np.linspace(20,80,31)

    #These are the base frequencies for the simulated beams (i.e. the simulated beams
    #had no more frequencies within this range). Assuming the same frequencies are used 
    #for future simulated beams, nothing here will have to be changed. If it is a different
    #range of frequencies, you will have to change this and alter the code coming up. 
    
    #The idea was to use one beam (i.e. the 20MHz beam) to approximate the beam for 20.5, 21 and 21.5 MHz (since 22MHz was the next simulated beam)
    #to have a plot that is smoother and has more points. If you'd like to remove this, 
    #simply use the test block above as a reference for a single frequency and choose your 
    #frequencies. 
    
    vismat_real=np.zeros([len(vis_real),len(nu)])
    vismat_im=np.zeros([len(vis_im),len(nu)])
    
    jcur = 0
    for i in range(len(nu_base)):
        
        icur = 0
        
        print(i,nu_base[i])
        
        beam = np.load(path_to_beam +  f'beam_{nu_base[i]}mhz_{pol}.npy')
        alms = healpy.map2alm(beam)
        
        while icur < 4: 
            
            if jcur == 121: 
                break
            
            print(i,nu_base[i],jcur,nu[jcur])
            
            alms_gsm = np.load(path_to_gsm + f'gsm_map_{nu_base[i]}mhz_2016_alm.npy')
            
            
            lamda= 300/nu[jcur]
            myphase= np.exp(2*np.pi*1J*np.dot(xyz,bl)/lamda)
            alms_real=healpy.map2alm(np.real(myphase)*beam)
            vec_real=collapse_alm_prod(alms_real,alms_gsm)
            alms_im=healpy.map2alm(np.imag(myphase)*beam)
            vec_im=collapse_alm_prod(alms_im,alms_gsm)

            vis_real,phi_real=beam_integral_fullday_from_collapsed(vec_real,alms[0])
            vis_im,phi_im=beam_integral_fullday_from_collapsed(vec_im,alms[0])
            vismat_real[:,jcur]=vis_real
            vismat_im[:,jcur]=vis_im
            
            jcur +=1
            icur+=1
        
    angles = np.arctan2(vismat_im, vismat_real)
    
    return angles, vismat_real, vismat_im


# In[ ]:


#Now, using the functions: 

vismat_real, vismat_im = phase_plot(2**10, np.linspace(20,80,121), path_to_beam, path_to_gsm, pol = 'NS' )

#Phase plot
angles = np.arctan2(vismat_im, vismat_real)
plt.imshow(angles,cmap = 'coolwarm', extent = [20,80,24,0]) #extent scales the axis correctly
plt.axis('auto')
plt.colorbar()
plt.show()

#Amplitude Plot
amplitude = np.log(np.sqrt(vismat_real**2 + vismat_imag**2))
plt.imshow(amplitude, cmap = 'coolwarm', extent = [20,80,24,0]) #extent scales the axis correctly
plt.axis('auto')
plt.colorbar()
plt.show()


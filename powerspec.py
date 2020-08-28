#!/usr/bin/env python
# coding: utf-8

# In[6]:


import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import scipy


# In[12]:


def get_u_th(ant1, ant2, nside):
    npix = hp.nside2npix(nside)
    th, phi = hp.pix2ang(nside, np.arange(npix))

    xyz = np.array([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)]).T

    re=40e6/2/np.pi

    ant1_xyz=np.asarray([np.cos(ant1[0]*np.pi/180)*np.cos(ant1[1]*np.pi/180),np.cos(ant1[0]*np.pi/180)*np.sin(ant1[1]*np.pi/180),np.sin(ant1[0]*np.pi/180)])
    ant2_xyz=np.asarray([np.cos(ant2[0]*np.pi/180)*np.cos(ant2[1]*np.pi/180),np.cos(ant2[0]*np.pi/180)*np.sin(ant2[1]*np.pi/180),np.sin(ant2[0]*np.pi/180)])

    ddist=(ant1_xyz-ant2_xyz)*re
    
    return np.dot(xyz, ddist)

def pwr_spec_vs_L(freqs, nside, pol, path_to_beam ):
    
    #Change this if using different antennas
    u_th = get_u_th(ant1 = [-46.88687, 37.82053 ], ant2 = [-46.886733,37.81902 ], nside = 2**10)
    
    for i in freqs: 
        
        
        #And this
        ant1 = [-46.88687, 37.82053 ]
        ant2 = [-46.886733,37.81902 ]

        re=40e6/2/np.pi

        ant1_xyz=np.asarray([np.cos(ant1[0]*np.pi/180)*np.cos(ant1[1]*np.pi/180),np.cos(ant1[0]*np.pi/180)*np.sin(ant1[1]*np.pi/180),np.sin(ant1[0]*np.pi/180)])
        ant2_xyz=np.asarray([np.cos(ant2[0]*np.pi/180)*np.cos(ant2[1]*np.pi/180),np.cos(ant2[0]*np.pi/180)*np.sin(ant2[1]*np.pi/180),np.sin(ant2[0]*np.pi/180)])


        th,phi=hp.pix2ang(nside,np.arange(hp.nside2npix(nside)))
        xyz=np.empty([len(th),3])
        xyz[:,0]=np.sin(th)*np.cos(phi)
        xyz[:,1]=np.sin(th)*np.sin(phi)
        xyz[:,2]=np.cos(th)

        
        #Generate phi
        lamda = scipy.constants.c/(i*1e6)
        phi = np.exp(1j*2*np.pi*u_th/lamda)
        
        #Load in beam. Make sure that it is centered properly otherwise plot will look weird.
        beam_XY = np.load(path_to_beam + f'beam_{i}mhz_{pol}.npy')
        
        #Get beam*phi
        beam_phi_XY = beam_XY*phi
        
        #Power Spec
        power_spec = hp.anafast(np.real(beam_phi_XY)) #Imaginary component ends up being the same. Theory says 
        #it shouldn't. May want to look into this more. 
        
        #Plot
        plt.plot(np.arange(power_spec.size), (power_spec), label = f'{i}MHz')
    
    plt.xlabel('L')
    plt.ylabel('Power Spectrum')
    plt.title(f'{pol} pol')
    plt.legend()
    plt.xlim(0,200)
    #plt.savefig('powerspec')
    plt.show()
    
    return 


# In[13]:


pwr_spec_vs_L(np.array([30.]), 2**10, 'EW', '/home/mattias/Desktop/McGill/Research Summer/2020/pygsm/beam_70mhz/')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import scipy


# In[3]:


def healpy_beam(beam_dict, NESW, healpy_nside=2**10, site_latitude=-46.88694):
    import healpy
    """ Converts a beam simulation dictionary into HealPix format.

    Given an input dictionary `beam_dict` containing the raw beam simulation and
    associated spherical coordinates, generates a new dictionary is generated in
    which the beam amplitudes and spherical coordinates are adapted to the
    HealPix format with pixelization set by `healpy_nside`.

    Args:
        beam_dict: a dictionary containing a raw beam simulation.
        healpy_nside: an integer specipying the HealPix pixelization.
        site_latitude: the latitute of the instrument associated with the beam.

    Returns:
        A dictionary containing the sampled HealPix beam amplitudes and
        associated spherical coordinates for each frequency channel (in MHz). A
        typical output returned by this function would have the following
        structure.

        {
        'theta': numpy.array,
        'phi': numpy.array,
        20: numpy.array,
        22: numpy.array,
        ...,
        198: numpy.array,
        200: numpy.array,
        'normalization': numpy.array,
        }
    """

    # Initializes the dictionary which will hold the HealPy version of the beam.
    healpy_beam_dict = {}

    # Extracts the frequencies for which beams are available in `beam_dict`.
    frequencies = [key for key in beam_dict.keys() if isinstance(key, float)]
    n_freq = len(frequencies)
    
    # Initializes a HealPy pixelization and associated spherical coordinates.
    healpy_npix = healpy.nside2npix(healpy_nside)
    healpy_theta, healpy_phi = healpy.pix2ang(healpy_nside,
                                              np.arange(healpy_npix))

    # Stores spherical coordinates in `healpy_beam_dict`.
    healpy_beam_dict['theta'] = healpy_theta
    healpy_beam_dict['phi'] = healpy_phi

    # SciPy 2D interpolation forces us to do proceed in chunks of constant
    # coordinate `healpy_theta`. Below we find the indices at which
    # `healpy_theta` changes.
    indices = np.where(np.diff(healpy_theta) != 0)[0]
    indices = np.append(0, indices + 1)

    # Initializes the NumPy array which will contain the normalization factor
    # for each beam.
    beam_norms = np.zeros(n_freq)
    
    # Loops over the different frequencies for which the beam has been
    # simulated.
    for i, frequency in enumerate(frequencies):

        # Computes the actual beam from the information contained in
        # `beam_dict`.
        beam = 10**(beam_dict[frequency]/10)
        
        
        # Interpolates beam.
        beam_interp = interp2d(beam_dict['theta'],
                                           beam_dict['phi'],
                                           beam,
                                           kind='cubic',
                                           fill_value=0)
        
        # Initializes `healpy_beam`, the HealPy version of the beam.
        healpy_beam = np.zeros(len(healpy_theta))

        # Constructs the HealPy beam.
        for j in range(np.int(len(indices)/2) + 2):
            start = indices[j]
            end = indices[j+1]
            healpy_beam[start:end] = beam_interp(healpy_theta[start],
                                             healpy_phi[start:end])[:,0]

        # Fills `beam_norms` with the appropriate normalization factors for
        # each HealPy beam.
        beam_norms[i] = np.sqrt(np.sum(healpy_beam**2))
        
        # Rotates and stores the the HealPy beam in the `healpy_beam_dict` under
        # the appropriate frequency entry.
        beam_rotation = healpy.rotator.Rotator([NESW, 0, 90 - site_latitude])
        healpy_beam = beam_rotation.rotate_map_pixel(healpy_beam/beam_norms[i])
        healpy_beam_dict[frequency] = healpy_beam

    # Adds the beam normalizations as a separate entry in `heapy_beam_dict`.
    healpy_beam_dict['normalization'] = beam_norms

    # Returns the HealPy version of the beam in a dictionary format.
    return healpy_beam_dict


# In[ ]:


#Load in beam, change this as needed 
beamfile = np.load('/home/mattias/Desktop/McGill/Research Summer/2020/pygsm/beam_70mhz/beam.npy')
freqs = np.load('/home/mattias/Desktop/McGill/Research Summer/2020/pygsm/beam_70mhz/freqs.npy')
thetas = np.load('/home/mattias/Desktop/McGill/Research Summer/2020/pygsm/beam_70mhz/thmat.npy')
phis = np.load('/home/mattias/Desktop/McGill/Research Summer/2020/pygsm/beam_70mhz/phimat.npy')


# In[ ]:


#Convert beam files to dictionary, thank you to Kelly for this section.  
beam_dict = {}
beam_dict['theta'] = thetas[0]
beam_dict['phi'] = phis[:,0]

for i in range(len(freqs)):
    f = (freqs[i])
    beam_dict[f] = beamfile[:,:,i]


# In[ ]:


#Note that the following will generate a dictionary containing a beam for each frequency in the file. 
#It would be a good idea to save each individual beam to your drive to call later so you don't need to generate 
#the beam each time since this code does take a while to run. 
NS_healpy_beam = healpy_beam(beam_dict, NESW = 0)
EW_healpy_beam = healpy_beam(beam_dict, NESW = 90)


# In[ ]:


#You can access a beam for a specific frequency by doing:

#70mhz beam at 40mhz
beam70_40mhz_NS = NS_healpy_beam[40.]
beam70_40mhz_EW = EW_healpy_beam[40.]

hp.mollview(beam70_40mhz_NS)
hp.mollview(beam70_40mhz_EW)


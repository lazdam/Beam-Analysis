#!/usr/bin/env python
# coding: utf-8

# In[3]:


import healpy as hp
import numpy as np

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

def beam_integral(beam,map):
    return np.sum(beam*map)/np.sum(beam)

def rotate_beam_phi(beam,phi):
    nside=healpy.npix2nside(len(beam))
    alm_beam=healpy.map2alm(beam)
    mvec=get_mvec(alm_beam)
    alm_rot=alm_beam*np.exp(1J*mvec*phi)
    beam_rot=healpy.alm2map(alm_rot,nside)
    return beam_rot

def beam_integral_rotate(beam,map,phi):
    beam_rot=rotate_beam_phi(beam,phi)
    return beam_integral(beam_rot,map)

def beam_integral_alms(beam_alms,map_alms):
    lmax=nalm2lmax(len(beam_alms))
    tot1=np.sum(np.conj(beam_alms[:lmax])*map_alms[:lmax])
    tot2=np.sum(np.conj(beam_alms[lmax:])*map_alms[lmax:])
    mynorm=beam_alms[0]*np.sqrt(4*np.pi)
    #return np.real(tot1+2*tot2)/np.real(beam_alms[0])
    return np.real((tot1+2*tot2)/mynorm)

def beam_integral_alms_rot(beam_alms,map_alms,phi):
    mvec=get_mvec(beam_alms)
    beam_alms_rot=beam_alms*np.exp(1J*mvec*phi)
    return beam_integral_alms(beam_alms_rot,map_alms)

def collapse_alm_prod(alms1,alms2, padded = 0):
    lmax=nalm2lmax(len(alms1))
    vec=np.zeros(lmax+1,dtype='complex')
    icur=0
    for m in range(lmax):
        nm=lmax-m
        vec[m]=np.sum((alms1[icur:icur+nm])*np.conj(alms2[icur:icur+nm]))
        icur=icur+nm
        
    y = np.zeros(padded)
    
    vec = np.hstack([vec, y])
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
    ans=(vecft/mynorm)
    ans=ans*len(ans) #need this normalization factor to undo the one built into the ifft
    phi = np.linspace(0,2*np.pi,len(ans))
    #phi=2*np.pi*np.arange(len(ans))/len(ans)
    return ans,phi


# In[4]:



def amp_vs_time(beam_tup, nside, freq, u = np.array([115,0,0]), padded = 0):
    
    c = 299792458 #m/s
    
    beam, xyz, a_0_0, lmax = beam_tup
    
    alm_map = generate_map2alm(freq, nside)

    lamda = c/(freq*1e6)

    phi = np.exp(1j*2*np.pi*np.dot(xyz, u)/lamda)

    rippled_beam = beam*phi

    #Split real and imag components
    real_beam = np.real(rippled_beam)
    imag_beam = np.imag(rippled_beam)

    alm_real = hp.map2alm(real_beam)
    alm_imag = hp.map2alm(imag_beam)

    #Collapse matrices for iFFT
    z_m_real = collapse_alm_prod(alm_real, alm_map, padded )
    z_m_imag = collapse_alm_prod(alm_imag, alm_map, padded )

    #Apply iFFT
    ans_real, phi_real = beam_integral_fullday_from_collapsed(z_m_real,a_0_0)
    ans_imag, phi_imag = beam_integral_fullday_from_collapsed(z_m_imag,a_0_0)

    
    return ans_real, ans_imag, phi_real 
    
def std_vs_nside_plot(nside_arr, freqs, u = np.array([115,0,0])):
    
    #Get pixel locations
    nside_max = max(nside_arr)
    th, phi = hp.pix2ang(nside_max, np.arange(hp.nside2npix(nside_max)))
    xyz = np.array([np.sin(th)*np.sin(phi), np.sin(th)*np.cos(phi), np.cos(th)]).T
    u = np.array([115,0,0])
    u_th = np.dot(xyz, u)
    
    
    #Initialize matrix that will contain std(nside)
    
    mtx = np.zeros([len(freqs), len(nside_arr) - 1], dtype = 'float')
    
    icur = 0
    for freq in freqs: 
        
        signals = {}
        
        #Generate beams and signals
        for nside in nside_arr: 
            
            beam  = generate_beam(nside, 50, np.array([0,1,0]))

            if nside != nside_max: 
                signals[nside] = amp_vs_time_plot(beam, nside, freq, padded = (3*(nside_max)) - (3*(nside)))

            else: 
                signals[nside] = amp_vs_time_plot(beam, nside, freq, padded = 0)
            
        #determine Standard Dev. 
        r_std_signal = np.std(np.real(signals[nside_max][0]), ddof = 1)
        i_std_signal = np.std(np.real(signals[nside_max][1]), ddof = 1)

        #std of difference
        std_diff_real = {}
        std_diff_imag = {}
        mean_std_diff = []

        for i in nside_arr[0:-1]: 
            std_diff_real[i] = np.std(np.real(signals[nside_max][0] - signals[i][0]))
            std_diff_imag[i] = np.std(np.real(signals[nside_max][1] - signals[i][1]))
            mean_std_diff.append((std_diff_imag[i] + std_diff_real[i])/2)

    
        mtx[icur, :] = mean_std_diff/((r_std_signal + i_std_signal)/2)
        icur +=1
    
    return mtx 
        


# In[ ]:


mtx = std_vs_nside_plot(np.array([32, 64, 128, 256, 512, 1024]), np.linspace(10,200,191))
for i in range(1,5):
    plt.plot(np.log2(np.array([32, 64, 128, 256, 512])), np.log(mtx[i*40]), label = f'{i*40}MHz', ls = '--', marker = 'o')
plt.legend()
plt.xlabel('log2(nside)')
plt.ylabel('log[σ(ΔT)/σ(Signal)]')
plt.title('Ratio of σs vs. nside')
plt.show()


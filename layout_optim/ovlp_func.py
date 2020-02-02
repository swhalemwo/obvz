import numpy as np

# # pythran export pythran_ovlp(float list, int list)

#pythran export pythran_ovlp(float[], int list)
def pythran_ovlp(pos, row_order):
    """see if rectangles (indicated by 4 corner points) overlap, 
    has to be called with just 1D of points
    """
    
    # pos = np.array(pos_list)

    nbr_nds = int(pos.shape[0]/4)
    nbr_pts = pos.shape[0]

    d = pos[:,np.newaxis] - pos[np.newaxis,:]
    
    d_rshp = d.reshape((int((nbr_pts**2)/4), 4))
    
    d_rshp_rord = d_rshp[row_order]

    d_rord2 = d_rshp_rord.reshape((nbr_nds, nbr_nds, 16))
    
    d_min = np.min(d_rord2, axis = 2)
    d_max = np.max(d_rord2, axis = 2)
    
    d_shrt = np.min(np.abs(d_rord2), axis = 2)
    
    d_ovlp = d_min * d_max
    
    # is to find things that overlap at corners to avoid dividing by 0 errors
    # zeros = np.where(d_ovlp == 0.0)
    # d_ovlp.put(zeros, [1]*2*len(zeros[0]))
    # np.put(d_ovlp, zeros, [0]*len(zeros))

    d_ovlp[d_ovlp == 0] = 1

    # numpy.nonzero(x==0)[0]
    
    # what's the idea here? 
    # probably about setting sings properly
    d_ovlp2 = (np.abs(d_ovlp)/d_ovlp)*(-1)

    # d_ovlp2 = (d_ovlp2 + 1.0)/2.0

    d_ovlp2[d_ovlp2 == -1] = 0
    # d_ovlp3 = np.copy(d_ovlp2)

    # d_ovlp3.clip(0.0, 1.0)

    return d_ovlp, d_ovlp2, d_shrt
    # return d

# ftemplate-depth can be increased, but still error
# try just parts? 



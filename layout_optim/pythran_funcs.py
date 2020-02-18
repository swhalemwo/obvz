import numpy as np
from time import time


# # pythran export pythran_ovlp(float list, int list)
# #pythran export pythran_ovlp(float[:,:], int list)

#pythran export pythran_ovlp(float64[:,:])
def pythran_ovlp(pos):
    """see if rectangles (indicated by 2 dimension extremes) overlap, 
    has to be called with just 1D of points
    """
    
    # pos = np.array(range(12))

    nbr_nds = int(pos.shape[0]/2)
    nbr_pts = pos.shape[0]

    d = pos[:,np.newaxis] - pos[np.newaxis,:]
    
    h, w = d.shape
    # nrows/ncols : size of each cell -> 2
    nrows = ncols = 2


    res_ar = d.reshape(int(h/nrows), nrows, -1, ncols)
    res_ar2 = res_ar.swapaxes(1,2)
    # think it should be 3d to collapse 3rd dimension to still allow multiplication
    # into 3,3,4? -> nbr_nds, nbr_nds, 4
    

    d_rord2 = res_ar2.reshape(nbr_nds,nbr_nds,4)

    
    d_min = np.min(d_rord2, axis = 2)
    d_max = np.max(d_rord2, axis = 2)
    
    d_shrt = np.min(np.abs(d_rord2), axis = 2)
    
    d_ovlp = d_min * d_max
    
    # is to find things that overlap at corners to avoid dividing by 0 errors
    # zeros = np.where(d_ovlp == 0.0)
    # d_ovlp.put(zeros, [1]*2*len(zeros[0]))
    # np.put(d_ovlp, zeros, [0]*len(zeros))

    d_ovlp[d_ovlp == 0] = 1

    # what's the idea here? 
    # probably about setting sings properly
    d_ovlp2 = (np.abs(d_ovlp)/d_ovlp)*(-1)

    d_ovlp2[d_ovlp2 == -1] = 0
    
    return d_ovlp2, d_shrt
    # return d

#pythran export pythran_ovlp_cbn(float32[:,:], int list)
def pythran_ovlp_cbn(pos, row_order):
    """see if rectangles (indicated by 2 dimension extremes) overlap, 
    has to be called with just 1D of points
    combines all stuff: float32, sliced extremes, manual row_ordering
    """
    
    nbr_nds = int(pos.shape[0]/2)
    nbr_pts = pos.shape[0]

    d = pos[:,np.newaxis] - pos[np.newaxis,:]
    
    
    d_rshp = d.reshape((int(2*nbr_nds**2), 2))
    d_rshp_rord = d_rshp[row_order]
    d_rord2 = d_rshp_rord.reshape((nbr_nds, nbr_nds, 4))
    
    d_min = np.min(d_rord2, axis = 2)
    d_max = np.max(d_rord2, axis = 2)
    
    d_shrt = np.min(np.abs(d_rord2), axis = 2)
    
    d_ovlp = d_min * d_max
    
    d_ovlp[d_ovlp == 0] = 1

    # what's the idea here? 
    # probably about setting sings properly
    d_ovlp2 = (np.abs(d_ovlp)/d_ovlp)*(-1)

    d_ovlp2[d_ovlp2 == -1] = 0
    
    return d_ovlp2, d_shrt
    # return d




#pythran export pythran_ovlp_32(float32[:,:])
def pythran_ovlp_32(pos):
    """see if rectangles (indicated by 2 dimension extremes) overlap, 
    has to be called with just 1D of points
    """
    
    # pos = np.array(range(12))

    nbr_nds = int(pos.shape[0]/2)
    nbr_pts = pos.shape[0]

    d = pos[:,np.newaxis] - pos[np.newaxis,:]
    
    h, w = d.shape
    # nrows/ncols : size of each cell -> 2
    nrows = ncols = 2


    res_ar = d.reshape(int(h/nrows), nrows, -1, ncols)
    res_ar2 = res_ar.swapaxes(1,2)
    # think it should be 3d to collapse 3rd dimension to still allow multiplication
    # into 3,3,4? -> nbr_nds, nbr_nds, 4
    

    d_rord2 = res_ar2.reshape(nbr_nds,nbr_nds,4)

    
    d_min = np.min(d_rord2, axis = 2)
    d_max = np.max(d_rord2, axis = 2)
    
    d_shrt = np.min(np.abs(d_rord2), axis = 2)
    
    d_ovlp = d_min * d_max
    
    # is to find things that overlap at corners to avoid dividing by 0 errors
    # zeros = np.where(d_ovlp == 0.0)
    # d_ovlp.put(zeros, [1]*2*len(zeros[0]))
    # np.put(d_ovlp, zeros, [0]*len(zeros))

    d_ovlp[d_ovlp == 0] = 1

    # what's the idea here? 
    # probably about setting sings properly
    d_ovlp2 = (np.abs(d_ovlp)/d_ovlp)*(-1)

    d_ovlp2[d_ovlp2 == -1] = 0
    
    return d_ovlp2, d_shrt
    # return d





#pythran export pythran_ovlp_old(float64[:,:], int list)
def pythran_ovlp_old(pos, row_order):
    """see if rectangles (indicated by 4 corner points) overlap, 
    has to be called with just 1D of points
    """
    
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

    # what's the idea here? 
    # probably about setting sings properly
    d_ovlp2 = (np.abs(d_ovlp)/d_ovlp)*(-1)

    d_ovlp2[d_ovlp2 == -1] = 0
    
    return d_ovlp2, d_shrt
    # return d


# pythran export pythran_ovlp_old_split(float32[:,:])
def pythran_ovlp_old_split(pos):
    """see if rectangles (indicated by 4 corner points) overlap, 
    has to be called with just 1D of points
    """
    
    nbr_nds = int(pos.shape[0]/4)
    nbr_pts = pos.shape[0]

    d = pos[:,np.newaxis] - pos[np.newaxis,:]
    
    # d_rshp = d.reshape((int((nbr_pts**2)/4), 4))
    # d_rshp_rord = d_rshp[row_order]
    # d_rord2 = d_rshp_rord.reshape((nbr_nds, nbr_nds, 16))
    
    h, w = d.shape
    # nrows/ncols : size of each cell -> 2
    nrows = ncols = 4

    res_ar = d.reshape(int(h/nrows), nrows, -1, ncols)
    res_ar2 = res_ar.swapaxes(1,2)
    # think it should be 3d to collapse 3rd dimension to still allow multiplication
    # into 3,3,4? -> nbr_nds, nbr_nds, 4

    d_rord2 = res_ar2.reshape(nbr_nds,nbr_nds,16)


    d_min = np.min(d_rord2, axis = 2)
    d_max = np.max(d_rord2, axis = 2)
    
    d_shrt = np.min(np.abs(d_rord2), axis = 2)
    
    d_ovlp = d_min * d_max
    
    # is to find things that overlap at corners to avoid dividing by 0 errors
    # zeros = np.where(d_ovlp == 0.0)
    # d_ovlp.put(zeros, [1]*2*len(zeros[0]))
    # np.put(d_ovlp, zeros, [0]*len(zeros))

    d_ovlp[d_ovlp == 0] = 1

    # what's the idea here? 
    # probably about setting sings properly
    d_ovlp2 = (np.abs(d_ovlp)/d_ovlp)*(-1)

    d_ovlp2[d_ovlp2 == -1] = 0
    
    return d_ovlp2, d_shrt




# ftemplate-depth can be increased, but still error
# try just parts? 

# # #pythran export nest_test()
# def nest_test():
#     test_ar = np.array(range(100))
#     test_order = list(range(2500))
    
#     res = pythran_ovlp(test_ar, test_order)
#     return(res)





#pythran export pythran_dist(float[:,:], int, int)
def pythran_dist(pos, nbr_nds, nbr_pts):


    x_pts = pos[:,0][0::2].copy()
    x_ovlp, dx_min = pythran_ovlp(x_pts)

    y_pts = pos[:,1][0::2].copy()
    y_ovlp, dy_min = pythran_ovlp(y_pts)

    both_ovlp = x_ovlp * y_ovlp
    x_ovlp2 = x_ovlp - both_ovlp
    y_ovlp2 = y_ovlp - both_ovlp

    none_ovlp = np.ones((nbr_nds, nbr_nds)) - both_ovlp - x_ovlp2 - y_ovlp2

    # # also have to get the point distances for none_ovlp (then shortest)
    delta_pts = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_pts = np.sqrt(np.sum(delta_pts**2, axis = -1))


    # dist_rshp = dist_pts.reshape((int((nbr_pts**2)/4), 4))
    # dist_rshp_rord = dist_rshp[row_order]
    # dist_rord = dist_rshp_rord.reshape((nbr_nds, nbr_nds, 16))
    # min_pt_dists = np.min(dist_rord, axis = 2)

    # x = np.vstack([pos, pos2]).T
    # delta_x = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # dist_pts = np.sqrt(np.sum(delta_x**2, axis = -1))

    
    h, w = dist_pts.shape
    # nrows/ncols : size of each cell -> now 4
    nrows = ncols = 4

    res_ar = dist_pts.reshape(int(h/nrows), nrows, -1, ncols)
    res_ar2 = res_ar.swapaxes(1,2)

    # now has to be collapsed into 16
    dist_rord = res_ar2.reshape(nbr_nds,nbr_nds,16)
    min_pt_dists = np.min(dist_rord, axis = 2)



    distance = (x_ovlp2 * dy_min) + (y_ovlp2 * dx_min) + (both_ovlp * 1) + (none_ovlp * min_pt_dists)
    np.clip(distance, 1, None, out = distance)

    # return dist_pts
    both_ovlp_cnt = both_ovlp.sum()


    # print('pythran dist values start')
    # print('x_ovlp: ', x_ovlp)
    # print('y_ovlp: ', y_ovlp)

    # print('both_ovlp: ', both_ovlp)
    # print('both_ovlp_cnt: ', both_ovlp_cnt)
    # print('pythran dist values end')

    return distance, both_ovlp_cnt



#pythran export pythran_dist_old(float[:,:], int list, int, int)
def pythran_dist_old(pos, row_order, nbr_nds, nbr_pts):


    x_pts = pos[:,0].copy()
    x_ovlp, dx_min = pythran_ovlp_old(x_pts, row_order)

    y_pts = pos[:,1].copy()
    y_ovlp, dy_min = pythran_ovlp_old(y_pts, row_order)

    both_ovlp = x_ovlp * y_ovlp
    x_ovlp2 = x_ovlp - both_ovlp
    y_ovlp2 = y_ovlp - both_ovlp

    none_ovlp = np.ones((nbr_nds, nbr_nds)) - both_ovlp - x_ovlp2 - y_ovlp2

    # # also have to get the point distances for none_ovlp (then shortest)
    delta_pts = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_pts = np.sqrt(np.sum(delta_pts**2, axis = -1))


    dist_rshp = dist_pts.reshape((int((nbr_pts**2)/4), 4))
    dist_rshp_rord = dist_rshp[row_order]
    dist_rord = dist_rshp_rord.reshape((nbr_nds, nbr_nds, 16))
    min_pt_dists = np.min(dist_rord, axis = 2)

    distance = (x_ovlp2 * dy_min) + (y_ovlp2 * dx_min) + (both_ovlp * 1) + (none_ovlp * min_pt_dists)
    np.clip(distance, 1, None, out = distance)

    # return dist_pts
    both_ovlp_cnt = both_ovlp.sum()


    # print('pythran dist values start')
    # print('x_ovlp: ', x_ovlp)
    # print('y_ovlp: ', y_ovlp)

    # print('both_ovlp: ', both_ovlp)
    # print('both_ovlp_cnt: ', both_ovlp_cnt)
    # print('pythran dist values end')

    return distance, both_ovlp_cnt




# # speed test by only focusing on min/max 

# from time import time
# from random import sample, shuffle

# ar1 = np.array(sample(range(100), k = 40))
# ar2 = np.array(sample(range(100), k = 80))

# row_order1 = list(range(int((40*40)/4)))
# shuffle(row_order1)

# row_order2 = list(range(int((80*80)/4)))
# shuffle(row_order2)

# t1 = time()
# for i in range(600):
#     pythran_ovlp(ar1, row_order1)
# t2 = time()
# t2-t1


# t3 = time()
# for i in range(600):
#     pythran_ovlp(ar2, row_order2)
# t4 = time()
# t4-t3


# # ---------- speed test end ---------

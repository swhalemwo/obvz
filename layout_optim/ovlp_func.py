import numpy as np



# # pythran export pythran_ovlp(float list, int list)

#pythran export pythran_ovlp(float[], int list)
def pythran_ovlp(pos, row_order):
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

# ftemplate-depth can be increased, but still error
# try just parts? 

# # #pythran export nest_test()
# def nest_test():
#     test_ar = np.array(range(100))
#     test_order = list(range(2500))
    
#     res = pythran_ovlp(test_ar, test_order)
#     return(res)
    



#pythran export pythran_dist(float[:,:], int list, int, int)
def pythran_dist(pos, row_order, nbr_nds, nbr_pts):

    x_pts = pos[:,0].copy()
    x_ovlp, dx_min = pythran_ovlp(x_pts, row_order)

    y_pts = pos[:,1].copy()
    y_ovlp, dy_min = pythran_ovlp(y_pts, row_order)

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
    return distance

 
# #pythran export ovlp_wrap(float[:,:], int list)
# def ovlp_wrap(pos, row_order):
    
#     # x_pts = pos

#     x_pts = pos[:,0].copy()
#     # x_ovlp, dx_min = pythran_ovlp(np.copy(pos[:,0]), row_order)
#     x_ovlp, dx_min = pythran_ovlp(x_pts, row_order)
#     return x_ovlp




# # #pythran export pythran_itrtr(float, float, int, int list, float, float)


# #pythran export pythran_itrtr(float64[:,:], float64[:,:], int[:,:], int list, float64[:,:], float64[:,:])
# # def pythran_itrtr(pos, pos_nds_init, A, row_order, dim_ar, delta_pos):
#     t = 12
    
#     ctr = 100
#     dt = t/ctr

#     # pos_nds = np.copy(pos_nds_init)

#     k = 20
#     height = width = 800
#     nbr_nds = pos_nds_init.shape[0]
#     nbr_pts = pos.shape[0]

    
#     while True:    
        
#         # try initing pos_nds only in loop
#         if ctr == 100:
#             pos_nds = np.copy(pos_nds_init)


#         delta_nds = pos_nds[:, np.newaxis, :] - pos_nds[np.newaxis, :, :]

#         # .astype('float64')
#         # .astype('float64')
        
#         # x_ovlp, dx_min = pythran_ovlp(np.copy(pos[:,0]), row_order)
#         # y_ovlp, dy_min = pythran_ovlp(np.copy(pos[:,1]), row_order)
        
#         # both_ovlp = x_ovlp * y_ovlp
#         # x_ovlp2 = x_ovlp - both_ovlp
#         # y_ovlp2 = y_ovlp - both_ovlp

#         # none_ovlp = np.ones((nbr_nds, nbr_nds)) - both_ovlp - x_ovlp2 - y_ovlp2

#         # # # also have to get the point distances for none_ovlp (then shortest)
#         # delta_pts = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
#         # # dist_pts = np.linalg.norm(delta_pts, axis=-1)
        
#         # # ----- linalg replacement
        
#         # # x = np.sqrt(np.sum(delta_pts**2, axis = -1))
        
#         # # x[0:4,0:4]
#         # # dist_pts[0:4,0:4]
        
#         # dist_pts = np.sqrt(np.sum(delta_pts**2, axis = -1))

#         # # ----- linalg replacement
        

#         # # dist_rshp = np.reshape(dist_pts, (int((nbr_pts**2)/4), 4))
#         # dist_rshp = dist_pts.reshape((int((nbr_pts**2)/4), 4))
        
#         # dist_rshp_rord = dist_rshp[row_order]
#         # # dist_rord = np.reshape(dist_rshp_rord, (nbr_nds, nbr_nds, 16))
#         # dist_rord = dist_rshp_rord.reshape((nbr_nds, nbr_nds, 16))
        
#         # min_pt_dists = np.min(dist_rord, axis = 2)

#         # distance = (x_ovlp2 * dy_min) + (y_ovlp2 * dx_min) + (both_ovlp * 1) + (none_ovlp * min_pt_dists)
#         # np.clip(distance, 1, None, out = distance)

#         # # displacement = np.einsum('ijk,ij->ik',
#         # #                          delta_nds,
#         # #                          (k * k / distance**2) - A * distance / k)
        
#         force_ar = (k * k / distance**2) - A * distance / k
#         displacement = (delta_nds * force_ar[:, :, None]).sum(axis=1)


#         # ------------- repellant borders
#         # have to be changed for pythran
        
#         # displacement[:,0] = displacement[:,0] + displacement[:,0]

#         dispx1 = np.copy(displacement[:,0]) + (k*10)**2/(pos_nds[:,0] - dim_ar[:,0]/2)**2
#         dispx2 = dispx1 - (k*10)**2/(width - (pos_nds[:,0] + dim_ar[:,0]/2))**2

#         dispy1 = np.copy(displacement[:,1]) + (k*10)**2/(pos_nds[:,1] - dim_ar[:,1]/2)**2
#         dispy2 = dispy1 - (k*10)**2/(height - (pos_nds[:,1] + dim_ar[:,1]/2))**2

#         # dispx2
#         # dispx2[0:8]
#         # displacement[0:8,0]

#         displacement = np.concatenate([dispx2[:,None], dispy2[:,None]], axis = 1)
#         # displacement = np.vstack([dispx2, dispy2]).T
#         # --------------- displacement change done

#         # length = np.linalg.norm(displacement, axis=-1)
        
#         length = np.sqrt(np.sum(displacement**2, axis = -1))
        
#         # maybe length array gets messed up here? 
#         # length = np.where(length < 0.01, 0.1, length)
        
#         # delta_pos2[0:5,:]
#         # delta_pos[0:5,:]

#         delta_pos -= delta_pos
#         # len_ar = len_ar

#         # len_ar += t/length
        
#         len_ar = t/length
#         # len_ar (1D) can be calculated, but not used for calculating displacements
        
#         len_ar3 = np.concatenate([len_ar[:,None], len_ar[:,None]], axis = 1)
#         # try concating length array


#         # len_ar = np.vstack([t/length]*2).T
#         # len_ar2 = len_ar.reshape((nbr_nds, 2))
#         # len_ar2 (2D) can be constructed too, but not used either
#         # reshaping len_ar2 doesn't make a difference
    
        
#         # displacement2 = displacement.reshape((nbr_nds, 2))
#         # reshape displacement? can be generated
#         # reshaped displacements don't work tho

#         # displacement = displacement.astype('float')

#         # delta_pos += displacement * len_ar[:,None]
#         # delta_pos += displacement2 * len_ar
#         # delta_pos += displacement * len_ar3
#         # difference between updating (+=) and re-assigning (=) delta pos? nope neither works
#         # reshaped displacement and non-reshaped length ar? nope, 
        
#         # get error about pos_nds although not crucially relevant
        

#         # ---------- update node positions
#         # pos_nds += delta_pos
        
#         # assert pos_nds.shape == delta_pos.shape

#         # pos_nds = pos_nds + delta_pos
        
#         # delta_pos =  np.array([range(nbr_nds), range(nbr_nds)]).T
#         # pos_nds = np.array([range(nbr_nds), range(nbr_nds)]).T

    
#         # pos_nds2 = np.concatenate([pos_nds[:,0,None], pos_nds[:,1,None]], axis =1)

#         # pos_nds = np.add(pos_nds, delta_pos)
#         # pos_nds = pos_nds2

#         # pos_nds = np.copy(pos_nds2)
        
#         # up0 = pos_nds[:,0] + delta_pos[:,0]
#         # up1 = pos_nds[:,1] + delta_pos[:,1]

#         # pos_nds = np.concatenate([up0[:,None], up1[:,None]], axis = 1)

#         # pos_nds = pos_nds + delta_pos

#         # pos_nds = pos_nds2


#         # scale delta_pos to corner points
#         # delta_pos_xtnd = np.reshape(np.hstack([delta_pos]*4), (nbr_pts, 2))
#         # delta_pos_xtnd = np.hstack([delta_pos]*4).reshape((nbr_pts, 2))
#         # pos += delta_pos_xtnd

#         ctr -= 1
#         if ctr == 0:
#             pos_nds -= pos_nds
#             pos_nds += len_ar3
#             # pos = displacement
#             break

#         t -= dt
        
#     # return len_ar3, displacement
#     return pos_nds, pos

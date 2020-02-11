import numpy as np
from pythran_funcs import pythran_dist


# #pythran export ovlp_wrap(float[:,:], int list)
# def ovlp_wrap(pos, row_order):
    
#     # x_pts = pos

#     x_pts = pos[:,0].copy()
#     # x_ovlp, dx_min = pythran_ovlp(np.copy(pos[:,0]), row_order)
#     x_ovlp, dx_min = pythran_ovlp(x_pts, row_order)
#     return x_ovlp




#pythran export pythran_itrtr(float64[:,:], float64[:,:], float[:,:], int list, float64[:,:], float, int, float, float, float, float)
def pythran_itrtr(pos, pos_nds, A, row_order, dim_ar, t, def_itr, rep_nd_brd_start, k, height, width):
    """calculates new positions given 
    - vertex/node positions (pos, pos_nds), 
    - connectivity matrix A
    - dimension array dim_ar
    - initial temperature t
    - minimum definite number of iterations def_itr
    - percentage of last section of def_itr when to start making node borders repellant
    - ideal distance between nodes k
    - height and width of canvas
    """
    # t = 12
    
    # def_itr = 100
    dt = t/def_itr

    # rep_nd_brd_start = 0.3
    # k = 20
    # height = width = 1000

    nbr_nds = pos_nds.shape[0]
    nbr_pts = pos.shape[0]

    ctr = 0
    
    while True:    

        # print('pos_nds: ', pos_nds)
        # print(t)

        delta_nds = pos_nds[:, np.newaxis, :] - pos_nds[np.newaxis, :, :]
        
        # calculate distances based on whether to use node borders or not
        if t < dt * def_itr * rep_nd_brd_start:
            
            # print('repellant node borders now')
            distance, both_ovlp_cnt = pythran_dist(pos, row_order, nbr_nds, nbr_pts)
            # print('both_ovlp: ', both_ovlp_cnt)
            
            
        else: 
            distance = np.sqrt(np.sum(delta_nds**2, axis = -1))
            # print('nodes as points')
            distance[distance == 0] = 1
            # print('distance: ', distance)


        force_ar = (k * k / distance**2) - A * distance / k
        # print('A: ', A)
        # print('k: ', k)
        # print('force_ar : ', force_ar)
        # print('delta_nds: ', delta_nds)
        displacement = (delta_nds * force_ar[:, :, None]).sum(axis=1)


        # ------------- repellant borders, could be fictionalized

        dispx1 = np.copy(displacement[:,0]) + (k*10)**2/(pos_nds[:,0] - dim_ar[:,0]/2)**2
        dispx2 = dispx1 - (k*10)**2/(width - (pos_nds[:,0] + dim_ar[:,0]/2))**2

        dispy1 = np.copy(displacement[:,1]) + (k*10)**2/(pos_nds[:,1] - dim_ar[:,1]/2)**2
        dispy2 = dispy1 - (k*10)**2/(height - (pos_nds[:,1] + dim_ar[:,1]/2))**2

        displacement = np.concatenate([dispx2[:,None], dispy2[:,None]], axis = 1)

        # -------- gravity
        
        # cntr = np.array([2,2])
        
        # cntr_vec_abs = np.abs(cntr_vec)
        # cntr_vec_abs[cntr_vec_abs == 0] = 1

        # cntr_vec/cntr_vec_abs
        # nope that removes the angle
        # need sum to be 1


        center_vec = center - pos_nds

        sum_vec = np.abs(np.sum(center_vec, axis =1))
        
        gravity_vec = (center_vec/sum_vec[:,None])*grav_multiplier
        displacement = displacement + gravity_vec

        # need vector to center

        # --------------- displacement change done

        length = np.sqrt(np.sum(displacement**2, axis = -1))
        length = np.where(length < 0.01, 0.1, length)

        len_ar = t/length
        delta_pos = displacement * len_ar[:,None]

        # ---------- update node positions
        # print('pos_nds v2: ', pos_nds)
        # print('delta_pos: ', delta_pos)
        
        pos_nds += delta_pos
        
        # print('pos_nds v3: ', pos_nds)

        delta_pos_xtnd = np.hstack([delta_pos]*4).reshape((nbr_pts, 2))
        pos += delta_pos_xtnd
        
        # reduce temperature in first phase (no node borders)
        # # reduce temp in second phase if nodes don't overlap
        
        ctr +=1
        if c == 500:
            break

        if t > (dt * def_itr * rep_nd_brd_start):
            t -= dt

        else: 

            if both_ovlp_cnt == nbr_nds:
                t -= dt

        if t < 0:
            break
        
    return pos_nds, pos, ctr

# * scratch

# x_ovlp, dx_min = pythran_ovlp(np.copy(pos[:,0]), row_order)
# y_ovlp, dy_min = pythran_ovlp(np.copy(pos[:,1]), row_order)

# both_ovlp = x_ovlp * y_ovlp
# x_ovlp2 = x_ovlp - both_ovlp
# y_ovlp2 = y_ovlp - both_ovlp

# none_ovlp = np.ones((nbr_nds, nbr_nds)) - both_ovlp - x_ovlp2 - y_ovlp2

# # # also have to get the point distances for none_ovlp (then shortest)
# delta_pts = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
# # dist_pts = np.linalg.norm(delta_pts, axis=-1)

# # ----- linalg replacement

# # x = np.sqrt(np.sum(delta_pts**2, axis = -1))

# # x[0:4,0:4]
# # dist_pts[0:4,0:4]

# dist_pts = np.sqrt(np.sum(delta_pts**2, axis = -1))

# # ----- linalg replacement


# # dist_rshp = np.reshape(dist_pts, (int((nbr_pts**2)/4), 4))
# dist_rshp = dist_pts.reshape((int((nbr_pts**2)/4), 4))

# dist_rshp_rord = dist_rshp[row_order]
# # dist_rord = np.reshape(dist_rshp_rord, (nbr_nds, nbr_nds, 16))
# dist_rord = dist_rshp_rord.reshape((nbr_nds, nbr_nds, 16))

# min_pt_dists = np.min(dist_rord, axis = 2)

# distance = (x_ovlp2 * dy_min) + (y_ovlp2 * dx_min) + (both_ovlp * 1) + (none_ovlp * min_pt_dists)
# np.clip(distance, 1, None, out = distance)

# * delta pos tries
# len_ar3 = np.concatenate([len_ar[:,None], len_ar[:,None]], axis = 1)
# try concating length array


# len_ar = np.vstack([t/length]*2).T
# len_ar2 = len_ar.reshape((nbr_nds, 2))
# len_ar2 (2D) can be constructed too, but not used either
# reshaping len_ar2 doesn't make a difference


# displacement2 = displacement.reshape((nbr_nds, 2))
# reshape displacement? can be generated
# reshaped displacements don't work tho

# displacement = displacement.astype('float')

# delta_pos += displacement2 * len_ar
# delta_pos += displacement * len_ar3
# difference between updating (+=) and re-assigning (=) delta pos? nope neither works
# reshaped displacement and non-reshaped length ar? nope, 

# get error about pos_nds although not crucially relevant


# assert pos_nds.shape == delta_pos.shape

# pos_nds = pos_nds + delta_pos

# delta_pos =  np.array([range(nbr_nds), range(nbr_nds)]).T
# pos_nds = np.array([range(nbr_nds), range(nbr_nds)]).T


# pos_nds2 = np.concatenate([pos_nds[:,0,None], pos_nds[:,1,None]], axis =1)

# pos_nds = np.add(pos_nds, delta_pos)
# pos_nds = pos_nds2

# pos_nds = np.copy(pos_nds2)

# up0 = pos_nds[:,0] + delta_pos[:,0]
# up1 = pos_nds[:,1] + delta_pos[:,1]

# pos_nds = np.concatenate([up0[:,None], up1[:,None]], axis = 1)

# pos_nds = pos_nds + delta_pos

# pos_nds = pos_nds2


# scale delta_pos to corner points
# delta_pos_xtnd = np.reshape(np.hstack([delta_pos]*4), (nbr_pts, 2))

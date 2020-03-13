import numpy as np

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


#pythran export pythran_dist_cbn(float32[:,:], int list, int, int)
def pythran_dist_cbn(pos, row_order, nbr_nds, nbr_pts):


    x_pts = pos[:,0][0::2].copy()
    x_ovlp, dx_min = pythran_ovlp_cbn(x_pts, row_order)

    y_pts = pos[:,1][0::2].copy()
    y_ovlp, dy_min = pythran_ovlp_cbn(y_pts, row_order)

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
    


#pythran export pythran_itrtr_cbn(float32[:,:], float32[:,:], float[:,:], int list, float32[:,:], float, int, float, float, float, float, float)
def pythran_itrtr_cbn(pos,pos_nds, A,row_order, dim_ar, t,def_itr,rep_nd_brd_start, k,height, width, grav_multiplier):
    """calculates new positions given 
    - vertex/node positions (pos, pos_nds), 
    - connectivity matrix A
    - dimension array dim_ar
    - initial temperature t
    - minimum definite number of iterations def_itr
    - percentage of last section of def_itr when to start making node borders repellent
    - ideal distance between nodes k
    - height and width of canvas
    - gravity multiplier
    """


    dt = t/def_itr

    nbr_nds = pos_nds.shape[0]
    nbr_pts = pos.shape[0]

    max_iter = 500
    ctr = 0

    center = np.array((width/2, height/2))

    while True:    
        
        delta_nds = pos_nds[:, np.newaxis, :] - pos_nds[np.newaxis, :, :]
        
        # calculate distances based on whether to use node borders or not
        if t < dt * def_itr * rep_nd_brd_start:

            # print('repellant node borders now')
            distance, both_ovlp_cnt = pythran_dist_cbn(pos, row_order, nbr_nds, nbr_pts)
            
            
        else: 
            # print('nodes as points')
            distance = np.sqrt(np.sum(delta_nds**2, axis = -1))
            
            distance[distance == 0] = 1
            # print('distance: ', distance)
        

        force_ar = (k * k / distance**2) - A * distance / k
        displacement = (delta_nds * force_ar[:, :, None]).sum(axis=1)

        # ------------- repellant borders, could be functionalized
        # why do i not get division by 0 error here? 

        dispx1 = np.copy(displacement[:,0]) + (k*10)**2/(pos_nds[:,0] - dim_ar[:,0]/2)**2
        dispx2 = dispx1 - (k*10)**2/(width - (pos_nds[:,0] + dim_ar[:,0]/2))**2

        dispy1 = np.copy(displacement[:,1]) + (k*10)**2/(pos_nds[:,1] - dim_ar[:,1]/2)**2
        dispy2 = dispy1 - (k*10)**2/(height - (pos_nds[:,1] + dim_ar[:,1]/2))**2

        displacement = np.concatenate([dispx2[:,None], dispy2[:,None]], axis = 1)

        # -------- gravity

        center_vec = center - pos_nds

        sum_vec = np.abs(np.sum(center_vec, axis =1))
        # prevent division by 0 error
        sum_vec[sum_vec == 0] = 1
        
        gravity_vec = (center_vec/sum_vec[:,None])*grav_multiplier
        displacement = displacement + gravity_vec
        

        # --------------- delta calcs
        # what is happening here
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
        
        # debugging test
        # if math.isnan(pos[0][0]):
        #     break
        
        # max iterations limit
        ctr +=1
        
        if ctr == max_iter:
            break
        
        # see if any nodes violate boundaries

        canvas_boundaries_crossed = 0
        
        min_x = np.min(pos[:,0])
        max_x = np.max(pos[:,0])

        min_y = np.min(pos[:,1])
        max_y = np.max(pos[:,1])
        
        if min_x < 0 or min_y < 0 or max_x > width or max_y > height: 
            canvas_boundaries_crossed = 1

        # reduce temperature in first phase (no node borders)
        if t > (dt * def_itr * rep_nd_brd_start):
            t -= dt

        # reduce temp in second phase if nodes don't overlap and boundaries not 
        else: 
            if both_ovlp_cnt == nbr_nds and canvas_boundaries_crossed == 0:
                t -= dt

        if t < 0:
            
            break
        
    return pos_nds, pos, ctr

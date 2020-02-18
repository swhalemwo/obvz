import numpy as np
from pythran_funcs import pythran_dist, pythran_dist_old
from time import time




#pythran export pythran_itrtr(float64[:,:], float64[:,:], float[:,:], int list, float64[:,:], float, int, float, float, float, float, float)
def pythran_itrtr(pos, pos_nds, A, row_order, dim_ar, t, def_itr, rep_nd_brd_start, k, height, width, grav_multiplier):
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

    t_func_start = time()

    dt = t/def_itr

    nbr_nds = pos_nds.shape[0]
    nbr_pts = pos.shape[0]

    max_iter = 500
    ctr = 0

    center = np.array((width/2, height/2))
    # grav_multiplier = 10

    
    # t_loop_start = time()
    
    while True:    
        t_itr_start = time()
        
        delta_nds = pos_nds[:, np.newaxis, :] - pos_nds[np.newaxis, :, :]
        
        # calculate distances based on whether to use node borders or not
        if t < dt * def_itr * rep_nd_brd_start:

            # print('repellant node borders now')
            distance, both_ovlp_cnt = pythran_dist(pos, nbr_nds, nbr_pts)
            # print('both_ovlp: ', both_ovlp_cnt)
            
            
        else: 
            # print('nodes as points')
            distance = np.sqrt(np.sum(delta_nds**2, axis = -1))
            
            distance[distance == 0] = 1
            # print('distance: ', distance)
            t_last_point_calc_itr = time()
        
        # t_dist_calced = time()

        force_ar = (k * k / distance**2) - A * distance / k
        displacement = (delta_nds * force_ar[:, :, None]).sum(axis=1)

        # t_displacement_done = time()
        
        # ------------- repellant borders, could be functionalized
        # why do i not get division by 0 error here? 

        dispx1 = np.copy(displacement[:,0]) + (k*10)**2/(pos_nds[:,0] - dim_ar[:,0]/2)**2
        dispx2 = dispx1 - (k*10)**2/(width - (pos_nds[:,0] + dim_ar[:,0]/2))**2

        dispy1 = np.copy(displacement[:,1]) + (k*10)**2/(pos_nds[:,1] - dim_ar[:,1]/2)**2
        dispy2 = dispy1 - (k*10)**2/(height - (pos_nds[:,1] + dim_ar[:,1]/2))**2

        displacement = np.concatenate([dispx2[:,None], dispy2[:,None]], axis = 1)

        # t_repellent_borders_done = time()
        # -------- gravity

        center_vec = center - pos_nds

        sum_vec = np.abs(np.sum(center_vec, axis =1))
        # prevent division by 0 error
        sum_vec[sum_vec == 0] = 1
        
        gravity_vec = (center_vec/sum_vec[:,None])*grav_multiplier
        displacement = displacement + gravity_vec
        
        # t_grav_done = time()
        
        # --------------- delta calcs

        length = np.sqrt(np.sum(displacement**2, axis = -1))
        length = np.where(length < 0.01, 0.1, length)

        len_ar = t/length
        delta_pos = displacement * len_ar[:,None]
        
        # t_deltas_done = time()
        
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
        
        # t_updates_done = time()
        
        # t_itr_ttl = t_updates_done - t_itr_start
        # t_dist_calced_prd = t_dist_calced - t_itr_start
        # t_displacement_done_prd = t_displacement_done - t_dist_calced
        # t_repellent_borders_done_prd = t_repellent_borders_done - t_displacement_done
        # t_grav_done_prd = t_grav_done - t_repellent_borders_done
        # t_deltas_done_prd = t_deltas_done - t_grav_done
        # t_updates_done_prd = t_updates_done - t_deltas_done
        
        

        # print('-------------------------')
        # print('t_dist_calced_prd: ', t_dist_calced_prd, round(t_dist_calced_prd/t_itr_ttl,3))
        # print('t_displacement_done_prd: ', t_displacement_done_prd, round(t_displacement_done_prd/t_itr_ttl,3))
        # print('t_repellent_borders_done_prd: ', t_repellent_borders_done_prd, round(t_repellent_borders_done_prd/t_itr_ttl,3))
        # print('t_grav_done_prd: ', t_grav_done_prd, round(t_grav_done_prd/t_itr_ttl, 3))
        # print('t_deltas_done_prd: ', t_deltas_done_prd, round(t_deltas_done_prd/t_itr_ttl, 3))
        # print('t_updates_done_prd: ', t_updates_done_prd, round(t_updates_done_prd/t_itr_ttl, 3))
        # print(ctr)
        
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
            t_func_end = time()
            t_func_duration = t_func_end - t_func_start
            t_point_period = t_last_point_calc_itr - t_func_start
            t_border_period = t_func_end - t_last_point_calc_itr
            print('iterations required: ', ctr)
            print('point period part: ', t_point_period, t_point_period/t_func_duration)
            print('border period: ', t_border_period, t_border_period/t_func_duration)
            
            break
        
    return pos_nds, pos, ctr


#pythran export pythran_itrtr_old(float64[:,:], float64[:,:], float[:,:], int list, float64[:,:], float, int, float, float, float, float, float)
def pythran_itrtr_old(pos, pos_nds, A, row_order, dim_ar, t, def_itr, rep_nd_brd_start, k, height, width, grav_multiplier):
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

    t_func_start = time()

    dt = t/def_itr

    nbr_nds = pos_nds.shape[0]
    nbr_pts = pos.shape[0]

    max_iter = 500
    ctr = 0

    center = np.array((width/2, height/2))
    # grav_multiplier = 10

    
    t_loop_start = time()
    
    while True:    
        # t_itr_start = time()
        
        delta_nds = pos_nds[:, np.newaxis, :] - pos_nds[np.newaxis, :, :]
        
        # calculate distances based on whether to use node borders or not
        if t < dt * def_itr * rep_nd_brd_start:

            # print('repellant node borders now')
            distance, both_ovlp_cnt = pythran_dist_old(pos, row_order, nbr_nds, nbr_pts)
            # print('both_ovlp: ', both_ovlp_cnt)
            
            
        else: 
            # print('nodes as points')
            distance = np.sqrt(np.sum(delta_nds**2, axis = -1))
            
            distance[distance == 0] = 1
            # print('distance: ', distance)
            t_last_point_calc_itr = time()
        
        # t_dist_calced = time()

        force_ar = (k * k / distance**2) - A * distance / k
        displacement = (delta_nds * force_ar[:, :, None]).sum(axis=1)

        # t_displacement_done = time()
        
        # ------------- repellant borders, could be functionalized
        # why do i not get division by 0 error here? 

        dispx1 = np.copy(displacement[:,0]) + (k*10)**2/(pos_nds[:,0] - dim_ar[:,0]/2)**2
        dispx2 = dispx1 - (k*10)**2/(width - (pos_nds[:,0] + dim_ar[:,0]/2))**2

        dispy1 = np.copy(displacement[:,1]) + (k*10)**2/(pos_nds[:,1] - dim_ar[:,1]/2)**2
        dispy2 = dispy1 - (k*10)**2/(height - (pos_nds[:,1] + dim_ar[:,1]/2))**2

        displacement = np.concatenate([dispx2[:,None], dispy2[:,None]], axis = 1)

        # t_repellent_borders_done = time()
        # -------- gravity

        center_vec = center - pos_nds

        sum_vec = np.abs(np.sum(center_vec, axis =1))
        # prevent division by 0 error
        sum_vec[sum_vec == 0] = 1
        
        gravity_vec = (center_vec/sum_vec[:,None])*grav_multiplier
        displacement = displacement + gravity_vec
        
        # t_grav_done = time()
        
        # --------------- delta calcs

        length = np.sqrt(np.sum(displacement**2, axis = -1))
        length = np.where(length < 0.01, 0.1, length)

        len_ar = t/length
        delta_pos = displacement * len_ar[:,None]
        
        # t_deltas_done = time()
        
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
        
        # t_updates_done = time()
        
        # t_itr_ttl = t_updates_done - t_itr_start
        # t_dist_calced_prd = t_dist_calced - t_itr_start
        # t_displacement_done_prd = t_displacement_done - t_dist_calced
        # t_repellent_borders_done_prd = t_repellent_borders_done - t_displacement_done
        # t_grav_done_prd = t_grav_done - t_repellent_borders_done
        # t_deltas_done_prd = t_deltas_done - t_grav_done
        # t_updates_done_prd = t_updates_done - t_deltas_done
        
        

        # print('-------------------------')
        # print('t_dist_calced_prd: ', t_dist_calced_prd, round(t_dist_calced_prd/t_itr_ttl,3))
        # print('t_displacement_done_prd: ', t_displacement_done_prd, round(t_displacement_done_prd/t_itr_ttl,3))
        # print('t_repellent_borders_done_prd: ', t_repellent_borders_done_prd, round(t_repellent_borders_done_prd/t_itr_ttl,3))
        # print('t_grav_done_prd: ', t_grav_done_prd, round(t_grav_done_prd/t_itr_ttl, 3))
        # print('t_deltas_done_prd: ', t_deltas_done_prd, round(t_deltas_done_prd/t_itr_ttl, 3))
        # print('t_updates_done_prd: ', t_updates_done_prd, round(t_updates_done_prd/t_itr_ttl, 3))
        # print(ctr)
        
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
            
            t_func_end = time()
            t_func_duration = t_func_end - t_func_start
            t_point_period = t_last_point_calc_itr - t_func_start
            t_border_period = t_func_end - t_last_point_calc_itr
            
            print('iterations required: ', ctr)
            print('point period part: ', t_point_period, t_point_period/t_func_duration)
            print('border period: ', t_border_period, t_border_period/t_func_duration)
            
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


# #pythran export ovlp_wrap(float[:,:], int list)
# def ovlp_wrap(pos, row_order):
    
#     # x_pts = pos

#     x_pts = pos[:,0].copy()
#     # x_ovlp, dx_min = pythran_ovlp(np.copy(pos[:,0]), row_order)
#     x_ovlp, dx_min = pythran_ovlp(x_pts, row_order)
#     return x_ovlp

import numpy as np
import time
import repartition_given_polygon

def remove_repeating_tiles(vp_tiles):
    vp_tiles = list(vp_tiles)
    #print('VP tiles', vp_tiles)
    tiles = []
    aux_tiles = []
    for t in vp_tiles:
        #print('list t', list(t)[:4])
        if not list(t)[:4] in aux_tiles:
            tiles.append(list(t))
            aux_tiles.append(list(t)[:4])

    return np.asarray(tiles)


def divide_common_distinct(user_area, SUM_AREA, combinations, n, u, video_name, f, gamma):
    start_time = time.time()
    TOTAL_TILES = []
    TILES_USER = {}
    #print('comb:', user_area)
    for item in combinations:
        area = np.zeros((10, 20))
    
        area[SUM_AREA != item] = 0
        area[SUM_AREA == item] = 1
        
        #print(area)
        tiles = repartition_given_polygon.repartition_tiles(area, video_name, f, gamma, n)
        #print('TILES:', tiles)
        
        for t in tiles:
            TOTAL_TILES.append(t)
    TOTAL_TILES = remove_repeating_tiles(TOTAL_TILES)
    #print('total tiles:', TOTAL_TILES)
    
    for item in TOTAL_TILES:
        for user in user_area:
            #print(item[0], item[2])
            if user_area[user][int(item[0])][int(item[1])] == 1:
                #print('SEE IF TILES ARE BEING CHOOSEN:', item, user_area[user][int(item[0])][int(item[1])], user)
                if user not in TILES_USER.keys():
                    TILES_USER[user] = []
                TILES_USER[user].append(item)
                
    #print('tiles user:', TILES_USER)
        
    return TILES_USER, TOTAL_TILES


def find_new_tiles_to_request(vp_tiles, c_tiles, data_store_path, n, u, video_name, f, gamma):
    
    start_time = time.time()

    tiles_to_fetch = []
    mat_repartition = np.zeros((10, 20))

    # check the overlapping tiles with the c tiles
    for v_t in vp_tiles:
        vp_mat = np.zeros((10, 20))
        vp_mat[int(v_t[0]):int(v_t[2]), int(v_t[1]):int(v_t[3])] = 1
        # prev_tot_bt = np.sum(vp_mat)

        for c_t in c_tiles:
            cache_mat = np.zeros((10, 20))
            cache_mat[int(c_t[0]):int(c_t[2]), int(c_t[1]):int(c_t[3])] = 1

            # check intersection
            overlap = vp_mat + cache_mat

            vp_mat[overlap == 2] = 0

        # curr_tot_bt = np.sum(vp_mat)

        # if prev_tot_bt == curr_tot_bt:
        #     tiles_to_fetch.append(v_t)
        # else:
        #     mat_repartition += vp_mat
        mat_repartition += vp_mat

    mat_repartition[mat_repartition > 0] = 1

    if np.sum(mat_repartition)>0:

        # -------------- Part of VASTile implementation for furthe partitioning the tiles starts -------------- #
        # partition the rectangular polygons to perfect rectangles.
        new_tiles = repartition_given_polygon.repartition_tiles(mat_repartition, video_name, f, gamma, n)
        for t in new_tiles:
            tiles_to_fetch.append(t)
        # -------------- Part of VASTile implementation for furthe partitioning the tiles ends --------------- #

    stop_time = time.time()

    return tiles_to_fetch, stop_time - start_time

# This python script combine all the functional blocks with the improved version
# consider only the cached tiles of the previous users
# dynamic matrix A creation

import pandas as pd
import numpy as np
import os

import settings
import create_mat_D
import cal_cost_r
import cal_cost_e
import cal_cost_s
import run_glpk
import partition_remianing_area
import check_distribution
import further_partition_new_tiles
import warnings
from termcolor import colored
import itertools

warnings.filterwarnings('ignore') 

is_DEBUG = True



def join_tiles(cached_tiles):
    
    joined_matriz = None 

    for item in cached_tiles:

        if joined_matriz is not None:
            n_c = np.array([item]*len(cached_tiles[item]))
            #print(joined_matriz)
            #print(np.c_[cached_tiles[item], n_c])
            joined_matriz = np.vstack((joined_matriz, np.c_[cached_tiles[item], n_c]))  

        else:
            n_c = np.array([item]*len(cached_tiles[item]))
            joined_matriz = np.c_[cached_tiles[item], n_c]

    return joined_matriz

def split_tiles(cached_tiles):
    #print(cached_tiles)
    splited = {}
    
    for item in cached_tiles:
        if item[8] in splited:
            splited[int(item[8])] = np.vstack((splited[int(item[8])], item[:8]))
        else:
            splited[int(item[8])] = np.array([item[:8]])
    #print('SPLITED')
    #print(splited)
    return splited



def cache_replacement_by_capacity(cached_tiles, batch, BT_size, cache_list):
    capacity = 15
    total_size = 0
    cached_tiles = join_tiles(cached_tiles)
    #print(cached_tiles)
    cached_tiles = cached_tiles[np.lexsort((-cached_tiles[:, 4], -cached_tiles[:, 7]))]
    #print('this is the verification', cached_tiles[:, 8])
    id_before = np.unique(cached_tiles[:, 8])
    index = 0
    while total_size <= capacity and index != len(cached_tiles):
        #print(cached_tiles[index][8])
        #print(BT_size[1])
        total_size = total_size + size_of_tile(cached_tiles[index], BT_size[int(cached_tiles[index][8])])
        index = index + 1
        #print('this works')
        #print(cached_tiles[index])
    #print('TOTAL SIZE', total_size)
    cached_tiles = cached_tiles[:index-1]
    id_after = np.unique(cached_tiles[:, 8])
    
    remaining = [int(i) for i in id_before if i not in id_after]
    
    #print('REMAINING', remaining)
    
    cached_tiles = split_tiles(cached_tiles)
    for i in remaining:
        cache_list.remove(i)
    #print('this is the caches inside replacement')  
    #print(cached_tiles)
    return cached_tiles, cache_list
    
# ++++++++++++++++++++++++++++#
# cache replace only if the number of users are greater than 15
def cache_replace(cached_tiles, n):
    cached_tiles = join_tiles(cached_tiles)

    
    cache_user_margin = 15
    cache_back_usage = 5
    cache_removing_percentile = 0.5
    cache_expected_hit_prob_of_removing_tiles = 0.2

    #print(cached_tiles)
    ch_replace_range = cached_tiles[cached_tiles[:, 4] < (n - cache_back_usage)]
    ch_non_replace_range = cached_tiles[cached_tiles[:, 4] >= (n - cache_back_usage)]

    # find the indicies should be removed
    # hit_val = ch_replace_range[:,-1]
    # hit_val_prob = hit_val/np.sum(hit_val)
    #print(ch_replace_range)
    ch_replace_range = ch_replace_range[np.lexsort((-ch_replace_range[:, 4], -ch_replace_range[:, 7]))]
    # cut the last 20% of the tiles
    last_20_ind = int(len(ch_replace_range) * cache_removing_percentile)
    total_hits = np.sum(ch_replace_range[:, 7])
    prob_cachehit_las_20 = np.sum(ch_replace_range[-last_20_ind:, 7]) / total_hits
    if prob_cachehit_las_20 < cache_expected_hit_prob_of_removing_tiles:
        ch_replace_range = ch_replace_range[:int(len(ch_replace_range) * (1 - cache_removing_percentile)), :]
        # ch_replace_range = ch_replace_range[np.lexsort(ch_replace_range[:, 4])]
        ch_replace_range = ch_replace_range[ch_replace_range[:, 4].argsort()]
    cached_tiles = np.concatenate([ch_replace_range, ch_non_replace_range], axis=0)
    
    cached_tiles = split_tiles(cached_tiles)
    return cached_tiles


def update_hits_in_cached_tiles(hit_cts, cached_tiles):
    y = hit_cts[:, 6]
    for i in y:
        indices = np.argwhere(cached_tiles[:, 6] == i)
        cached_tiles[indices, 7] += 1
    return cached_tiles


# Taking a given set of tiles from a user has streamed
# find the tiles areaday has been cached and store the new tiles from CS to be cached.
# cached the tiles into the global list with the user n data
# This cached tile list is a Global list that all users can see
def fill_cached_tiles(tiles, n, first_batch, u, cached_tiles, hit_cts):
    #print('this runs', n, cached_tiles)
    if len(tiles) == 0:
        if len(hit_cts)!=0:
            cached_tiles = update_hits_in_cached_tiles(hit_cts, cached_tiles)
    else:
        # add last column indicating the tile requesting order and the user order
        if first_batch:
            last_ind = -1
        else:
            last_ind = cached_tiles[-1, 6]

        new_indices = np.arange(last_ind + 1, len(tiles) + last_ind + 1).reshape([-1, 1])
        #print(new_indices)
        hits = np.zeros(len(tiles)).reshape([-1, 1])
        #print('HITS\n\n', hits)
        #return
        request_order = np.repeat([n], tiles.shape[0]).reshape([-1, 1])
        #print(request_order)
        user_num = np.repeat([u], tiles.shape[0]).reshape([-1, 1])
        #print(user_num)
        new_arr = np.concatenate([tiles, request_order, user_num, new_indices, hits], axis=1)

        if first_batch:
            cached_tiles = np.asarray(new_arr)
        else:
            #print('cached', cached_tiles)
            #print('new_array', new_arr)
            cached_tiles = np.concatenate([cached_tiles, new_arr], axis=0)

    return cached_tiles


def store_all_cached_tiles(cached_tiles, n, storage_path):
    ct_sel = cached_tiles
    columns = ['l_l_m', 'l_l_n', 'u_r_m', 'u_r_n', 'n', 'u', 'id', 'hit']
    df = pd.DataFrame(columns=columns,
                      data=ct_sel)
    user_path = storage_path + '/user_' + str(n)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    df.to_csv(user_path + '/total_cts.csv', index=False)

    return

def calculate_intersection(DT_tiles):
    #print(DT_tiles)
    bts = np.zeros((10, 20)).astype(int)
    for i in DT_tiles:
        bts_for_user = np.zeros((10, 20))
        for t in DT_tiles[i]:
            bts[int(t[0]):int(t[2]), int(t[1]):int(t[3])] = bts[int(t[0]):int(t[2]), int(t[1]):int(t[3])] + 1

    return bts
    
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

def create_dict_of_mapping_areas(users):
    num_us = len(users)
    cont = 1
    dict_of_mapping_areas = {}
    
    for i in range(1, num_us+1):
        combinations = itertools.combinations(users, i)
        for c in combinations:
            #print(sorted(list(c)))
            dict_of_mapping_areas[tuple(sorted(list(c)))] = cont
            cont = cont + 1
            
    
    #print(dict_of_mapping_areas[(1,)])
    #print(dict_of_mapping_areas[(1,2)])
    return dict_of_mapping_areas         
    
def calculate_intersection_of_remaining_tiles(users, t_for_complete, batch, vid, chunk):
    
        #print('this is running')
        user_area = {}
        combinations = []
        for user in t_for_complete:
            user_area[user] = calculate_intersection({user:t_for_complete[user]})
        #print(user_area)
        
        SUM_AREA = np.zeros((10, 20)).astype(int)
        dict_of_mapping_areas = create_dict_of_mapping_areas(users)
        #print(dict_of_mapping_areas)
        for row in range(0, 10):
            for col in range(0,20):
                list_of_users = []
                for user in users:
                    if user_area[user][row][col] == 1:
                        list_of_users.append(user)
                if len(list_of_users) != 0:
                    code_occ = dict_of_mapping_areas[tuple(sorted(list_of_users))]
                    if code_occ not in combinations:
                        combinations.append(code_occ)
                    SUM_AREA[row][col] = code_occ
        #print('c',combinations)       
        #print(SUM_AREA)
        ########## Divide the DTs between commun and distinct ###########
        TILES_USER = partition_remianing_area.divide_common_distinct(user_area, SUM_AREA, combinations, batch, users, video_name=vid,f=chunk,gamma=1)
        #print('RESTES', TILES_USER[0], '\n\n\n')
        return TILES_USER
    
    
    
def read_DT_and_calculate_common_and_distinct_VP_area(vid, vid_DT_path_in, users, batch, chunk):
        DT_tiles = {}
        user_area = {}
        combinations = []
            
        # read the DT tiles of individual users
        for user in users:
            user_path = vid_DT_path_in + '/user_' + str(user) + '/chunk_' + str(chunk) + '.csv'
            DT_tiles[user] = remove_repeating_tiles(pd.read_csv(user_path).values[:, :-1])
            #########  apenas para visualizar as areas individuais
            print('###### USER ', user, 'VIEW POINT')
            user_area[user] = calculate_intersection({user:DT_tiles[user]})
            DRAW_AREA(user_area[user], users)
            
            
            
        SUM_AREA = calculate_intersection(DT_tiles)
        print('|--AREA COMUM E INDIVIDUAL--|')
        DRAW_AREA(SUM_AREA, users)  
        
        SUM_AREA = np.zeros((10, 20)).astype(int)
        dict_of_mapping_areas = create_dict_of_mapping_areas(users)
        #print(dict_of_mapping_areas)
        for row in range(0, 10):
            for col in range(0,20):
                list_of_users = []
                for user in users:
                    if user_area[user][row][col] == 1:
                        list_of_users.append(user)
                if len(list_of_users) != 0:
                    code_occ = dict_of_mapping_areas[tuple(sorted(list_of_users))]
                    if code_occ not in combinations:
                        combinations.append(code_occ)
                    SUM_AREA[row][col] = code_occ
                
        #print(SUM_AREA)
                
        ########## Divide the DTs between commun and distinct ###########
        TILES_USER = partition_remianing_area.divide_common_distinct(user_area, SUM_AREA, combinations, batch, users, video_name=vid,f=chunk,gamma=1)
        #print(TILES_USER)
        return TILES_USER


def record_solution(batch, total_bits_internet, total_bits_wireless, Num_BT_in, Num_BT_wi):
    if batch == 0:
        with open('output/op3.csv', 'w') as file:
            file.write('batch;Internet;Wireless;BT_Int;BT_Wir\n')
            file.write(str(batch) + ';' + str(total_bits_internet) + ';' + str(total_bits_wireless) + ';' + str(int(Num_BT_in)) + ';' + str(int(Num_BT_wi)) + '\n')
            
    else:
        with open('output/op3.csv', 'a') as file:
            file.write(str(batch) + ';' + str(total_bits_internet) + ';' + str(total_bits_wireless) + ';' + str(int(Num_BT_in)) + ';' + str(int(Num_BT_wi)) + '\n')
            
            
def record_cache(total):
    with open('output/cache_op3.txt', 'a') as file:
        file.write(str(total) + '\n')

def size_of_tile(tile, BT_size):
    #print(tile)
    index_bt = [(i*20) + (j+1) for i in range(int(tile[0]), int(tile[2])) for j in range(int(tile[1]), int(tile[3]))]
    size_bt = float('{0:.10f}'.format(BT_size[index_bt].sum() / 1000000))
    dt_size_mb = 0.432 * size_bt * size_bt + 0.306 * size_bt + 0.0025 #formula do artigo
    return dt_size_mb
    
def calculate_size(cached_tiles, BT_size):
    total = 0
    for tile in cached_tiles:
        dt_size_mb = size_of_tile(tile, BT_size)
        total = total + dt_size_mb
    return total
        
    
def PRINT_SOLUTION(users, SOLUTION_CS, SOLUTION_CACHE, BT_size, t):
    
    #print(SOLUTION_CS)
    #print('\n')
    #print(SOLUTION_CACHE)
    Total_bits_wi = 0
    Total_bits_in = 0
    
    Num_BT_wi = 0
    Num_BT_in = 0 
    
    list_of_tiles = []

    
    print("|------------------------INTERNET LINK------------------------|")
    for user in users:
        print('\t\t', 'USER: ', user)
        if user in SOLUTION_CS:
            for tile in SOLUTION_CS[user]:
                Num_BT_in = Num_BT_in + ((tile[2]-tile[0])*(tile[3]-tile[1]))
                
                #index_bt = [(i*20) + (j+1) for i in range(int(tile[0]), int(tile[2])) for j in range(int(tile[1]), int(tile[3]))]
                #size_bt = BT_size[index_bt].sum()
                size_bt = size_of_tile(tile, BT_size)
                print('\t\t', list(map(int, tile)), 'Tamanho em bits:',size_bt)

                string_code = ''
                for elem in tile:
                    string_code = string_code + '_' + str(int(elem))
                    #print(string_code)
                    
                if string_code not in list_of_tiles:
                    list_of_tiles.append(string_code)
                    Total_bits_in = Total_bits_in + size_bt

    print('\t\t\tTotal bits', Total_bits_in)

        
        
    print("|---------------------WIRELESS LINK---------------------------|")
    
    for item in users:

        print('\t\t', 'USER: ', item)
        if item in SOLUTION_CACHE:
            for tile in SOLUTION_CACHE[item]:
                Num_BT_wi = Num_BT_wi + ((tile[2]-tile[0])*(tile[3]-tile[1]))
                
                #index_bt = [(i*20) + (j+1) for i in range(int(tile[0]), int(tile[2])) for j in range(int(tile[1]), int(tile[3]))]
                #size_bt = BT_size[index_bt].sum()
                size_bt = size_of_tile(tile, BT_size)
                Total_bits_wi = Total_bits_wi + size_bt

                print('\t\t', list(map(int, tile)), 'Tamanho em bits:',size_bt)

    print('\t\t\tTotal bits', Total_bits_wi)

    print("|--------------------------------------------------------------|")
    
    record_solution(t, Total_bits_in, Total_bits_wi, Num_BT_in, Num_BT_wi)

def DRAW_AREA(area, users):
    count = 0
    count_lines = 0
    print(colored('     0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8  9', 'blue'))
    for x in np.nditer(area):
        if x == len(users):
            color = 'red'
        elif x == 0:
            color = 'white'
        else:
            color = 'green'
            
        if count == 19:
            print(colored(str(x), color)+'.]', end ="\n")
            count = 0
        else:
            if count == 0:
                print(colored(str(count_lines) + ' - ', 'blue'), end='')
                count_lines = count_lines + 1
                print('[' + colored(str(x)+'.', color), end =" ")
                count = count + 1

            else:
                count = count + 1
                print(colored(str(x)+'.', color), end =" ")
                
                
                
def DRAW_INDIVIDUAL(users, user, SOLUTION_CS, SOLUTION_CACHE):
    area = np.zeros((10, 20)).astype(int)
    #if user in SOLUTION_CS:
    #    for dt in SOLUTION_CS[user]:
    #        area[int(dt[0]):int(dt[2]), int(dt[1]):int(dt[3])] = area[int(dt[0]):int(dt[2]), int(dt[1]):int(dt[3])] + 1
            
    if user in SOLUTION_CACHE:
        for dt in SOLUTION_CACHE[user]:
            area[int(dt[0]):int(dt[2]), int(dt[1]):int(dt[3])] = area[int(dt[0]):int(dt[2]), int(dt[1]):int(dt[3])] + 1
            
    DRAW_AREA(area, users)
    
                
def DRAW_SOLUTION(users, SOLUTION_CS, SOLUTION_CACHE):
    for user in users:
        print('###### USER ', user, 'SOLUTION')
        DRAW_INDIVIDUAL(users, user, SOLUTION_CS, SOLUTION_CACHE)
        
    area = np.zeros((10, 20)).astype(int)
    
    #for item in SOLUTION_CS:
    #    for dt in SOLUTION_CS[item]:
    #        area[int(dt[0]):int(dt[2]), int(dt[1]):int(dt[3])] = area[int(dt[0]):int(dt[2]), int(dt[1]):int(dt[3])] + 1
            
    for item in SOLUTION_CACHE:
        for dt in SOLUTION_CACHE[item]:
            area[int(dt[0]):int(dt[2]), int(dt[1]):int(dt[3])] = area[int(dt[0]):int(dt[2]), int(dt[1]):int(dt[3])] + 1

    print('|---ÁREA COBERTA/SOLUÇÃO---|')
    DRAW_AREA(area, users)
    print('\n\n')
            

def calculate_total_size(TOTAL_TILES, BT_size):
    total = 0
    for tile in TOTAL_TILES:
        print(tile)
        print(TOTAL_TILES)
        size_bt = size_of_tile(tile, BT_size)
        total = total + size_bt
        
    return total
    
def find_index_for_all_BTs(TOTAL_TILES):
    total_index = []
    for tile in TOTAL_TILES:
        ind = [(i*20) + (j+1) for i in range(int(tile[0]), int(tile[2])) for j in range(int(tile[1]), int(tile[3]))]
        total_index.extend(ind)
        
    return total_index

def calculate_hits(index_of_requisited_BTs, index_of_attended_BTs, BT_size):
    total_len_req = 0
    total_siz_req = 0
    
    total_len_att = 0
    total_siz_att = 0
    
    for item in index_of_requisited_BTs:
        index_req = list(dict.fromkeys(index_of_requisited_BTs[item]))
        index_att = list(dict.fromkeys(index_of_attended_BTs[item]))
        index_att = list(set(index_att) & set(index_req))
        
        total_len_att = total_len_att + len(index_att)
        total_len_req = total_len_req + len(index_req)
        
        total_siz_att = total_siz_att + BT_size[item][index_att].sum()
        total_siz_req = total_siz_req + BT_size[item][index_req].sum()
        
        
        
    return total_siz_att, total_siz_req, total_len_att, total_len_req
        
    
    
    
    
def run_algo(vid, rand_set,
             w1, w2, w3,
             DT_path, BT_path, Data_store_path):
    print(vid)
    # array to store the time consumption for the key tasks of the process.
    t_for_complete = {}

    # read random sets for the users are in different order.
    random_sets = settings.rand_set
    cached_tiles = {}
    cache_list = []

    # define paths for inputs and outputs
    vid_DT_path_in = DT_path + '/' + vid
    BT_size_path = BT_path + '/' + vid


    # read the bt size for the given chunk
    BT_size = pd.read_csv(BT_size_path + '.csv').values

    for batch, users in enumerate(random_sets[rand_set]):
        
        index_of_attended_BTs = {}
        index_of_requisited_BTs = {}
        
        print('BATCH', batch)
        for chunk in users:
            
            index_of_attended_BTs[chunk] = []
            
            print('CHUNK:', chunk, 'USERS:', users[chunk])
            if chunk not in cache_list:
                cached_tiles[chunk] = []
                cache_list.append(chunk)
                chunk_first_time = True
            else:
                chunk_first_time = False
                
            
            data_store_path = Data_store_path + '/' + vid + '/Random_set' + str(rand_set) + '/chunk_' + str(chunk)

            if not os.path.exists(data_store_path):
                os.makedirs(data_store_path)
                
            #print('CHUNKS', chunk, users, batch)
        
        
            SOLUTION_CS = {}
            SOLUTION_CACHE = {}
            #print(batch, users)

            TILES_USER, TOTAL_TILES = read_DT_and_calculate_common_and_distinct_VP_area(vid, vid_DT_path_in, users[chunk], batch, chunk)
            
            index_of_requisited_BTs[chunk] = find_index_for_all_BTs(TOTAL_TILES)
            #print(index_of_requisited_BTs)
            #print('ALL TILES:\n', TILES_USER)

            # take all the tiles streamed by batch n=0 from the CS as the cached tiles. n==0 means the first batch
            if batch == 0:

                # Store common VP dynamic tiles
                tiles_to_be_cached = TOTAL_TILES
                #for item in DT_DISTINCT:
                #    if len(tiles_to_be_cached) == 0:
                #        tiles_to_be_cached = DT_DISTINCT[item]
                #    elif len(DT_DISTINCT[item]) != 0:
                #            tiles_to_be_cached = np.concatenate([tiles_to_be_cached, DT_DISTINCT[item]], axis=0)
                for user in users[chunk]:
                    SOLUTION_CS[user] = list(TILES_USER[user])
                    SOLUTION_CACHE[user] = list(TILES_USER[user])

                    
                cached_tiles[chunk] = fill_cached_tiles(tiles_to_be_cached, batch, chunk_first_time, -2, cached_tiles[chunk], hit_cts=[])
                store_all_cached_tiles(cached_tiles[chunk], batch, data_store_path)
                

                #print(cached_tiles[chunk])


            # run the algorithm for the batch n>=1
            else:
                for user in users[chunk]:
                    #print('This is the test')
                    #print(cached_tiles[chunk])
                    D, time_pre_D, overlapped_ct = create_mat_D.create_matrix_D(TILES_USER[user], batch,
                                                                                data_store_path,
                                                                                cached_tiles[chunk],
                                                                                ena_store=True)

                    #print(D)

                    if len(overlapped_ct) > 0:
                        # generate the cost r
                        cost_r, time_cost_r = cal_cost_r.generate_cost_r(TILES_USER[user], batch,
                                                                         data_store_path,
                                                                         overlapped_ct,
                                                                         ena_store=True)
                        # calculate the cost_e
                        cost_e, time_cost_e = cal_cost_e.generate_cost_e(overlapped_ct,
                                                                         BT_size[chunk, 1:],
                                                                         batch,
                                                                         data_store_path,
                                                                         ena_store=True)
                        # calculate the cost_s
                        cost_s, time_for_cost_s = cal_cost_s.generate_cost_s(TILES_USER[user],
                                                                             overlapped_ct,
                                                                             BT_size[chunk, 1:],
                                                                             batch,
                                                                             data_store_path,
                                                                             ena_store=True)


                        sel_ct, t_ilp_sol = run_glpk.get_ilp_based_sol(D,
                                                                       cost_r, cost_e, cost_s,
                                                                       batch,
                                                                       data_store_path,
                                                                       w1, w2, w3,
                                                                       overlapped_ct,
                                                                       0,          
                                                                       ena_store=True)

                        index_of_attended_BTs[chunk].extend(find_index_for_all_BTs(sel_ct))

                        SOLUTION_CACHE[user] = list(sel_ct[:, :4])

                        #print(SOLUTION_CACHE)

                        fetch_t, t_fetch_new_t = partition_remianing_area.find_new_tiles_to_request(TILES_USER[user], sel_ct,
                                                                                                    data_store_path, batch, user,
                                                                                                    video_name=vid,
                                                                                                    f=chunk,
                                                                                                    gamma=1)
                        fetch_t = remove_repeating_tiles(fetch_t)
                        # --------------Part of VASTile implementation for furthe partitioning the tiles starts -------------- #
                        # further partitioned the tiles
                        fetch_t_further_partitioned, t_fetch_new_furth_part_t = further_partition_new_tiles.partition_new_tiles(
                            fetch_t, data_store_path, batch, user, gamma=1, ena_store=True)
                        t_for_complete[user] = fetch_t_further_partitioned

                        # --------------Part of VASTile implementation for furthe partitioning the tiles ends -------------- #

                        # fill the cached hit analysis
                        if len(sel_ct) > 0:
                            cached_tiles[chunk] = fill_cached_tiles([], batch, chunk_first_time, user, cached_tiles[chunk],
                                                             hit_cts=sel_ct)
                        # update the catched tiles witht the newly fetched tiles
                        #if len(fetch_t) > 0:
                        #    cached_tiles = fill_cached_tiles(np.asarray(fetch_t_further_partitioned), batch, item, cached_tiles,
                        #                                     hit_cts=[])
                        #store_all_cached_tiles(cached_tiles, batch, data_store_path)


                        #if user in SOLUTION_CS:
                        #    SOLUTION_CS[user] = list(SOLUTION_CS[user]) + list(fetch_t_further_partitioned)

                        #else:
                        #    SOLUTION_CS[item] = list(fetch_t_further_partitioned)


                        #if item in SOLUTION_CACHE:
                        #    SOLUTION_CACHE[item] = list(SOLUTION_CACHE[item]) + list(fetch_t_further_partitioned)
                        #else:
                        #    SOLUTION_CACHE[item] = list(fetch_t_further_partitioned)
                        #print('teste3\n\n\n', SOLUTION_CACHE[item])

                    else:
                        further_partition_new_tiles.store_new_t(TILES_USER[user], data_store_path, batch, user)
                        run_glpk.store_sel_ct([], data_store_path, batch)
                        t_for_complete[user] = TILES_USER[user]
                        #cached_tiles = fill_cached_tiles(np.asarray(TILES_USER[user]), batch, user, cached_tiles, hit_cts=[])
                        #store_all_cached_tiles(cached_tiles, batch, data_store_path)

                        #if item in SOLUTION_CS:
                        #    SOLUTION_CS[item] = list(SOLUTION_CS[item]) + list(DT_DISTINCT[item])
                        #else:
                        #    SOLUTION_CS[item] = list(DT_DISTINCT[item])


                        #if item in SOLUTION_CACHE:
                        #    SOLUTION_CACHE[item] = list(SOLUTION_CACHE[item]) + list(DT_DISTINCT[item])
                        #else:
                        #    SOLUTION_CACHE[item] = list(DT_DISTINCT[item])

                TILES_USER, TOTAL_TILES = calculate_intersection_of_remaining_tiles(users[chunk], t_for_complete,batch, vid, chunk)
                    
                
                #print('COMPLETE:', TILES_USER)

                if len(TOTAL_TILES) != 0:
                    cached_tiles[chunk] = fill_cached_tiles(np.asarray(TOTAL_TILES), batch, chunk_first_time, -1, cached_tiles[chunk], hit_cts=[])
                    #cached_tiles[chunk] = fill_cached_tiles(np.asarray(TOTAL_TILES), 0, -1, cached_tiles[chunk], hit_cts=[])

                    for item in TILES_USER:

                        SOLUTION_CS[item] = list(TILES_USER[item])

                        if item in SOLUTION_CACHE:
                            SOLUTION_CACHE[item] = list(SOLUTION_CACHE[item]) + list(TILES_USER[item])
                        else:
                            SOLUTION_CACHE[item] = list(TILES_USER[item])

                store_all_cached_tiles(cached_tiles[chunk], batch, data_store_path)



            DRAW_SOLUTION(users[chunk], SOLUTION_CS, SOLUTION_CACHE)
            PRINT_SOLUTION(users[chunk], SOLUTION_CS, SOLUTION_CACHE, BT_size[chunk],batch)
            

            print('CACHED TILES')
            for chunk_ind in cached_tiles:#ok
                print('chunk:', chunk_ind, 'quantidade:', len(cached_tiles[chunk_ind]))
                #print(cached_tiles[c_entries])
                
            print('\n\n')
        
        #cached_tiles = cache_replace(cached_tiles, batch)
        cached_tiles, cache_list = cache_replacement_by_capacity(cached_tiles, batch, BT_size, cache_list)
        
        
        print(index_of_requisited_BTs)
        print(index_of_attended_BTs)
        t_siz_att, t_siz_req, t_len_att, t_len_req = calculate_hits(index_of_requisited_BTs, index_of_attended_BTs, BT_size)

        
        with open('output/hits_op3.txt', 'a') as file:
            file.write(str(batch)  + ' ' + str(t_siz_att) + ' ' + str(t_siz_req) + ' ' + str(t_len_att) + ' ' + str(t_len_req) + '\n')
        
        size_cache = 0
        for chunk_ind in cached_tiles:
            size_cache = size_cache + calculate_size(cached_tiles[chunk_ind], BT_size[chunk_ind])
        
        #print(cached_tiles)
        record_cache(size_cache)
    return


def run_opcash(vid, w1, w2, w3, bw_trace, DT_path, BT_path, Data_store_path):
    # Run different random set of user sequence.
    # maximum there are 3 randomized orders in settings.py
    for rand_set in range(1):
        #for chunk in range(1):
        run_algo(vid, rand_set,
                 w1, w2, w3,
                 DT_path, BT_path, Data_store_path)

    return
    

def main():
    work_dir = os.getcwd()
    DT_path = work_dir + '/Tile_info_DT'
    BT_path = work_dir + '/Tile_info_BT'
    Data_store_path = work_dir + '/Data_store_path'

    bw_trace = pd.read_csv(work_dir + '/BW_traces/4G.csv').values[6, :-1]
    bw_trace = bw_trace / 8

    w1 = 0.6
    w2 = 0.25
    w3 = 0.15

    vids = settings.vid_test

    for v, vid in enumerate(vids):
        run_opcash(vid, w1, w2, w3, bw_trace, DT_path, BT_path, Data_store_path)
        break

if __name__ == main():
    main()

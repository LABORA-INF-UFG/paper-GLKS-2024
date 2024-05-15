DATASET_1 = 1
DATASET_2 = 2
DATASET_3 = 3
DATASET_4 = 4
DATASET_5 = 5

HORI_TILES = 20
VERTI_TILES = 10

NUM_OF_FRAMES = 120
NUM_OF_USERS_TRAIN_CLOSED = 20
NUM_OF_USERS_TRAIN_OPEN = 10
NUM_OF_USERS_TEST_CLOSED = 20
NUM_OF_USERS_TEST_OPEN = 10
NUM_OF_TOT_USERS = 30

NUM_OF_USERS_APPROXI_THRESH_VALID = 17

# Tile coordinate indices after the initial MNC algorithm
L_L_M = 2
L_L_N = 3
U_R_M = 6
U_R_N = 7
AVG_PIXEL_INT_PER_TILE = 10
FRAME_NUM = 1
SAL_STATE = 12

# tile coordinates for the combined FOV, Buffer and OOV files
L_L_M_COMB = 0
L_L_N_COMB = 1
U_R_M_COMB = 2
U_R_N_COMB = 3
AVG_SAL_COMB = 4
SAL_STATE_COMB = 5

APPROXI_THRESH = 0.5
FINER_THRESH = 0.9

LEFT_ALIGNED = 0
CENTER_ALIGNED = 1
RIGHT_ALIGNED = 2

UPPER_ALIGNED = 0
CENTER_ALIGNED = 1
LOWER_ALIGNED = 2

FRAME_ERROR = -1
FRAME_OK = 1

TILE_W = 48
TILE_H = 54

FRAME_W = 960
FRAME_H = 540

RES_HD = [1920, 1080]
RES_4K = [3840, 2160]

# all the videos used
video_names = [
    "6_2",
    "6_4",
    "6_5",
    "6_8",
    "6_10",
    "6_11",
    "6_12",
    # "6_13",
    "6_17",
    "6_21",
    "6_25",
    "6_29",
    "ChariotRace_new",
    "Diving",
    "DrivingWith_new",
    "FootBall_new",
    "HogRider_new",
    "Kangarooisland",
    "Kitchen_new",
    "MagicShow_new",
    "PerlisPanel",
    "Rhinos2",
    "Rollercoaster1",
    "RollerCoster_new",
    "SFRsport",
    "SharkShipWreck_new",
    "Skiing_new",
    "Tahitisurf",
    "Timelapse",
    "WeirdAl"
]

# videos used for the train purposes and parameter decisioning
vid_train = ['6_11',
             '6_29',
             'DrivingWith_new',
             'HogRider_new',
             'Rollercoaster1',
             "6_2",
             '6_8',
             '6_25',
             'Kangarooisland',
             "Skiing_new",
             "6_4",
             "6_17",
             "SharkShipWreck_new",
             'Tahitisurf',
             "6_21",
             "SFRsport",
             "6_5",
             "6_12",
             "PerlisPanel",
             "Kitchen_new",
             ]

# videos used for the test purposes.
# data related to these videos are avilable in Github repo
vid_test = ['ChariotRace_new',
            'RollerCoster_new',
            'FootBall_new',
            "MagicShow_new",
            "WeirdAl",
            "6_10",
            "Rhinos2",
            "Timelapse",
            "Diving",

            ]

video_static_focus = [
    "6_4",
    "6_17",
    "SharkShipWreck_new",
    'Tahitisurf',
    "MagicShow_new"
]

video_moving_focus = [
    "6_10",
    "6_21",
    "Rhinos2",
    "SFRsport",
]

video_misc = [
    "6_5",
    "6_12",
    "Diving",
    "PerlisPanel",
    "Timelapse",
    "Kitchen_new",
    "WeirdAl",
]

def function_chunks(chunks, users_clusters, num_clusters):
    user_order = []
    for i in range(0,70):
        dict = {}
        index = i
        user_index = 0
        dict[index] = users_clusters[user_index]
        #print(dict)
        user_order.append(dict)
    

    c1_index = 70
    c2_index = 0
    while c1_index != 120:
        dict = {}
        dict[c1_index] = users_clusters[0]
        dict[c2_index] = users_clusters[1]
        c1_index = c1_index + 1
        c2_index = c2_index + 1

        user_order.append(dict) 
        
        
    while c2_index != 120:
        dict = {}
        dict[c2_index] = users_clusters[1]
        c2_index = c2_index + 1

        user_order.append(dict)  

    return user_order


#users_clusters = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12, 13, 14], [15,16,17],[18,19,20],[21,22,23],[24,25,26],[27,28,29]]
users_clusters = [[0,1,2],[3,4,5]]
chunks = 120

user_order = function_chunks(chunks, users_clusters, len(users_clusters))
''' 
user_order = [{1:[0,1,2]}, #T1
                
                {2:[0,1,2], 1:[3,4,5]}, #T2
               
                {3:[0,1,2], 2:[3,4,5], 1:[6,7,8]}, #T3

                {4:[0,1,2], 3:[3,4,5], 2:[6,7,8], 1:[9,10,11]}, #T4
    
                {5:[0,1,2], 4:[3,4,5], 3:[6,7,8], 2:[9,10,11], 1:[12,13,14]}, #T5

                {6:[0,1,2], 5:[3,4,5], 4:[6,7,8], 3:[9,10,11], 2:[12,13,14], 1:[15,16,17]}, #T6
                
                {7:[0,1,2], 6:[3,4,5], 5:[6,7,8], 4:[9,10,11], 3:[12,13,14], 2:[15,16,17], 1:[18,19,20]}, #T7
                
                {8:[0,1,2], 7:[3,4,5], 6:[6,7,8], 5:[9,10,11], 4:[12,13,14], 3:[15,16,17], 5:[18,19,20], 1:[21,22,23]}, #T8
                
                {9:[0,1,2], 8:[3,4,5], 7:[6,7,8], 6:[9,10,11], 5:[12,13,14], 4:[15,16,17], 3:[18,19,20], 2:[21,22,23], 1:[24,25,26]}, #T9
                
               {10:[0,1,2], 9:[3,4,5], 8:[6,7,8], 7:[9,10,11], 6:[12,13,14], 5:[15,16,17], 4:[18,19,20], 3:[21,22,23], 2:[24,25,26], 1:[25,26,27]}] #T10
'''

rand_set = [user_order]
video_riding = [
    '6_11',
    '6_29',
    'ChariotRace_new',
    'DrivingWith_new',
    'HogRider_new',
    'Rollercoaster1',
    'RollerCoster_new']

video_explore = [
    "6_2",
    '6_8',
    '6_25',
    'FootBall_new',
    'Kangarooisland',
    "Skiing_new",
]

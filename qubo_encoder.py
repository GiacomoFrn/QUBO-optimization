import encodings
import pandas as pd
from itertools import combinations

# dense encoding dictionary
dense_encoding_dict = {1: [0, 1], 2: [1, 0], 3: [1, 1], 13: [13]}

# produce qubo indexing from single camera request for dense encoding
def index_from_camera(camera, id):
    # retrive encoding information
    enc = dense_encoding_dict[camera]
    # stereo case
    if len(enc)==1:
        return [(1, [[id, enc[0]]])]
    # mono case
    out = []
    if enc[0] and enc[1] :
        out.append((1, [[id, 0], [id, 1]]))
    else:
        for j in range(2):
            if enc[j]:
                out.append((1, [[id, j]]))
            else:
                out.append((-1, [[id, 0], [id, 1]]))
    return out

# formulate qubo istance from photo request
def qubo_from_request(request, option, encoding):
    # get features
    id = request[0]
    value = request[1]
    mono = request[2]
    camera = request[3][option]

    # standard encoding
    if encoding == "standard":
        # get qubo
        rank = 1
        coeff = -value # minimize qubo -> maximize value
        indexes = [[id, camera]] # if mono else [id]
        qubo = [list([rank, coeff, indexes])]

    # dense encoding
    if encoding == "dense":
        # get indexing
        idx = index_from_camera(camera, id)

        # get qubos
        qubo = []
        for i in idx:
            rank = len(i[1])
            coeff = -value*i[0]
            indexes = i[1]
            qubo.append(list([rank, coeff, indexes]))

    return qubo
        

# formulate qubo instance from constraint
def qubo_from_constraint(constraint, option, coeff, encoding):
    # get features
    ids = constraint[0]
    restriction = constraint[1][option]

    # standard encoding
    if encoding == "standard":
        # get qubo
        rank = len(ids)
        coeff = coeff
        indexes = [[id, camera] for id, camera in zip(ids, restriction)]
        qubo = [list([rank, coeff, indexes])]
    
    # dense encoding
    if encoding == "dense":
        # compress idxs to a single idx
        idxs = []
        for i in range(len(restriction)):
            idxs.append(index_from_camera(restriction[i], ids[i]))
        ## number of resulting qubo terms
        while len(idxs)!=1:
            idxs_ = []
            #print(idxs)
            for i in idxs[0]:
                #print(i)
                for j in idxs[1]:
                    #print(j)
                    idxs_.append((i[0]*j[0], i[1]+j[1]))
                    #print(idxs_)
            idxs = [idxs_] + idxs[2:]
            #print(idxs)

        idx = idxs[0]
        
        # get qubo terms
        qubo = []    
        for i in idx:
            rank = len(i[1])
            coeff = coeff*i[0]
            indexes = i[1]
            qubo.append(list([rank, coeff, indexes]))
    
    return qubo
    

def qubo_from_data(photo_req_df, constraints_df, encoding):
    # create empty qubo dataframe
    qubo_df = pd.DataFrame({'rank':    pd.Series(dtype='int'),
                            'coeff':   pd.Series(dtype='int'),
                            'indexes': pd.Series(dtype='object')})

    k = 0

    # populate qubo dataframe from photo requests
    for i in range(len(photo_req_df)):
        l = len(photo_req_df.loc[i]['options'])
        for j in range(l):
            qubo_inst = qubo_from_request(photo_req_df.loc[i], j, encoding)
            for q in qubo_inst:
                qubo_df.loc[k] = q
                k = k + 1

    # penalties coefficient
    m = -1.1*min(qubo_df['coeff'])                                                                           

    if encoding=="standard":
        # add penalties to avoid taking the same photo multiple times with different cameras
        for i in range(len(photo_req_df)): 
            if len(photo_req_df.loc[i]['options'])>1 :
                for j, z in combinations(photo_req_df.loc[i]['options'], 2):
                    qubo_df.loc[k] = list([2, m, [[i, photo_req_df.loc[i]['options'][j-1]], [i, photo_req_df.loc[i]['options'][z-1]]]])
                    k = k + 1

    # populate qubo dataframe from constraints
    for i in range(len(constraints_df)):
        l = len(constraints_df.loc[i]['restrictions'])
        for j in range(l):
            qubo_inst = qubo_from_constraint(constraints_df.loc[i], j, m, encoding)
            for q in qubo_inst:
                qubo_df.loc[k] = q
                k = k + 1

    return qubo_df

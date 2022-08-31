import encodings
import pandas as pd
import numpy as np
from itertools import combinations
#######################################################

# Read spot file

########################################################################################################################

# process photo requests
def request_from_line(line):
    # split line
    l = line.split()

    # get features
    id = int(l[0])
    value = int(l[1])
    mono = False if (l[2] == '1' and int(l[3])>4) else True
    options = [int(o) for o in l[3::2]]

    return list([id, value, mono, options])

# process constraints
def costraint_from_line(line):
    # split line
    l = line.split()

    # get features
    n = int(l[0])
    ids = [int(i) for i in l[1:n+1]]
    restrictions = [[int(l[j]) for j in range(i,i+n)] for i in range(n+1, len(l), n)]

    return list([ids, restrictions])

# read spot file
def read_spot_file(spot_dir, spot_file):
    # create empty photo requests dataframe
    photo_req_df = pd.DataFrame({'id':      pd.Series(dtype='int'),
                                'value':   pd.Series(dtype='int'),
                                'mono':    pd.Series(dtype='bool'),
                                'options': pd.Series(dtype='object')})

    # create empty constraints dataframe
    constraints_df = pd.DataFrame({'ids':            pd.Series(dtype='object'),
                               'restrictions':   pd.Series(dtype='object')})

    # read test file
    with open(spot_dir+spot_file) as f:
        lines = f.readlines()

    # get ids and reqs lengths
    lens = [int(l.split()[0]) for l in lines if len(l.split())==1]

    # create photo requests dataframe
    for i in range(1,lens[0]+1):
        photo_req_df.loc[i-1] = request_from_line(lines[i])

    # create constraints dataframe
    for i in range(lens[0]+2, lens[0]+lens[1]+2):
        constraints_df.loc[i-(lens[0]+2)] = costraint_from_line(lines[i])

    return photo_req_df, constraints_df

########################################################################################################################

# Encode data dataframes into binary variables

########################################################################################################################

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
            coeff_ = coeff*i[0]
            indexes = i[1]
            qubo.append(list([rank, coeff_, indexes]))
    
    return qubo
    

def qubo_from_data(photo_req_df, constraints_df, encoding, penalty_coeff=1.1):
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
    m = -penalty_coeff*min(qubo_df['coeff'])                                                                           

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


########################################################################################################################

# Preprocess binary df (qubo_df)
#   - group by duplicates
#   - from indexes to variables

########################################################################################################################

def preprocess_indexes(indexes, join=False):
        # sort if more than one element
        if len(indexes)>1:
            indexes = sorted(indexes, key=lambda x: x[0]+x[1]/10) # hardcoded solution to consider both sublists elements

        # produce string output
        proc_indexes = [str(i[0])+'_'+str(i[1]) for i in indexes ]
        return ''.join(proc_indexes) if join else proc_indexes

def process_qubo_df(qubo_df):    
    # group by operation needs string object
    qubo_df['gb_indexes'] = qubo_df['indexes'].apply(preprocess_indexes, join=True)
    qubo_df_ = qubo_df.groupby(['gb_indexes'], as_index=False, sort=False).agg({'rank': 'first', 'coeff': 'sum', 'indexes': 'first'}).drop(['gb_indexes'], axis=1)

    # produce hash column for dictionary indexing
    qubo_df_['keys'] = qubo_df_['indexes'].apply(preprocess_indexes, join=False)

    # produce 1 to one dict between keys of the dataframe and indexes of the qubo matrix
    keys = np.array([k_i for key in qubo_df_['keys'] for k_i in key])
    keys = np.unique(keys)
    key_to_qubo_dict = { key:i for i,key in enumerate(keys) }

    # add column with variable indexing
    qubo_df_['variables'] = qubo_df_['keys'].apply(lambda k: [key_to_qubo_dict[k_i] for k_i in k])
    qubo_df_ = qubo_df_.drop(['indexes', 'keys'], axis=1)
    qubo_df_ ['variables'] = qubo_df_ ['variables'].apply(lambda x: np.array(x))

    return qubo_df_, key_to_qubo_dict

########################################################################################################################

# Reduce higher order terms

########################################################################################################################
def preprocess_variables(variables, join=False):
            # sort if more than one element
            if len(variables)>1:
                variables = sorted(variables) # hardcoded solution to consider both sublists elements

            # produce string output
            proc_variables = [str(i) for i in variables ]
            return '_'.join(proc_variables) if join else proc_variables

def reduce_higher_order_terms(qubo_df_, key_to_qubo_dict, method='boros'):
    if method=='boros':
        ## Boros algorithm to reduce higher order terms in quadratic ones

        # set M = 1 + sum(|c_i|), m = n
        M = sum(abs(qubo_df_['coeff']))+1
        #print(M)
        m = len(key_to_qubo_dict)
        #print(m)

        # while there exist a term with rank > 2 
        while max(qubo_df_['rank'])>2:
            # qubo to key dictionary
            qubo_to_key_dict = {v: k for k, v in key_to_qubo_dict.items()}

            qubo = []
            # select an higher order term
            ho_term = np.array(qubo_df_[qubo_df_['rank']>2]['variables'])[0]
            #print(ho_term)

            # choose two elements from it 
            ij = ho_term[:2]
            #print(ij)

            # select terms with the two elements
            mask = [all(x in var for x in ij) for var in qubo_df_['variables']]
            ho_df = qubo_df_[mask]
            #print(ho_df)

            # update key_to_qubo_dict with the new variable
            key_to_qubo_dict['_'.join([qubo_to_key_dict[v] for v in ij])] = m

            # update i,j term
            i = ho_df.index[ho_df['rank']==2].to_list()
            if len(i)!=0:
                #print("found bin term", i[0])
                #qubo_df_.loc[i[0], 'coeff'] = qubo_df_.loc[i[0], 'coeff'] + M
                #qubo_df_.loc[i[0], 'variables'] = np.array([m])
                qubo.append(list([2, qubo_df_.loc[i[0], 'coeff'] + M, np.array([v for v in ij])]))
                qubo_df_.drop(i[0], inplace=True)
                ho_df = ho_df.drop(i[0])
            else:
                qubo.append(list([2, M, np.array([v for v in ij])]))

            # create terms [i, m; j, m]
            qubo.append(list([2, -2*M, np.array([ij[0], m])]))
            qubo.append(list([2, -2*M, np.array([ij[1], m])]))

            # create term m
            qubo.append(list([1, 3*M, np.array([m])]))
            #print(qubo)

            # change variables where i,j appear to m
            for i in ho_df.index:
                #print(i)
                mask_ = np.array([var not in ij for var in qubo_df_.loc[i, 'variables']])
                #print(mask_)
                #print(qubo_df_.loc[i, "variables"])
                var = np.append(qubo_df_.loc[i, "variables"][mask_],[m])
                qubo.append(list([len(var), qubo_df_.loc[i, 'coeff'], var]))
                #qubo_df_.loc[i, 'variables'] = 
                #qubo_df_.loc[i, 'rank'] = len(qubo_df_.loc[i, 'variables'])
                qubo_df_.drop(i, inplace=True)

            # add new terms to qubo_df_
            #print(qubo)
            qubo_df_.reset_index(drop=True, inplace=True)
            #print(qubo_df_)
            l = len(qubo_df_)
            for i in range(len(qubo)):
                qubo_df_.loc[i+l] = qubo[i]

            # update m
            m = m + 1
    
    if method=='ishikawa':
        ## Ishikawa algorithm to reduce higher order terms in quadratic ones

        # current number of variables
        w = len(key_to_qubo_dict)

        # select higher order terms
        ho_terms = qubo_df_[qubo_df_['rank']>2]
        #print(len(ho_terms), "higher order terms found")

        # list for new terms
        qubo = []

        # reduce higher order terms
        for i in ho_terms.index:
            # coefficient of the higher order term
            a = qubo_df_.loc[i, 'coeff']
            # order of the higher order term
            d = qubo_df_.loc[i, 'rank']
            # distinguish a<0 and a>0
            if a<0:
                # create new ancillary variable
                key_to_qubo_dict[f'{i}_ho_term'] = w
                # append new terms
                for var in qubo_df_.loc[i, 'variables']:
                    qubo.append(list([2, a, np.array([w, var])]))
                qubo.append(list([1, -(d-1)*a, np.array([w])]))
                # delete ho_term
                qubo_df_.drop(i, inplace=True)
                # update w
                w = w + 1
            else:
                nd = int(np.floor((d-1)/2))
                for j in range(1, nd+1):
                    c_jd = 1 if (j==nd and d%2!=0) else 2
                    # create new ancillary variable
                    key_to_qubo_dict[f'{i}_ho_term_{j}'] = w
                    # append new terms
                    for var in qubo_df_.loc[i, 'variables']:
                        qubo.append(list([2, -a*c_jd, np.array([w, var])]))
                    qubo.append(list([1, a*(2*j*c_jd-1), np.array([w])]))
                    w = w + 1
                for j in range(d-1):
                    for k in range(j+1, d):
                        var_j = qubo_df_.loc[i, 'variables'][j]
                        var_k = qubo_df_.loc[i, 'variables'][k]
                        qubo.append(list([2, a, np.array([var_j, var_k])]))
                # delete ho_term
                qubo_df_.drop(i, inplace=True)

        # add new terms to qubo_df_
        #print(qubo)
        qubo_df_.reset_index(drop=True, inplace=True)
        #print(qubo_df_)
        l = len(qubo_df_)
        for i in range(len(qubo)):
            qubo_df_.loc[i+l] = qubo[i]

        # group terms with same indexing and sum coefficients
        ## group by operation needs string object
        qubo_df_['gb_variables'] = qubo_df_['variables'].apply(preprocess_variables, join=True)

        # gruop by qubo_df_ by gb_variables summing coefficients
        qubo_df_ = qubo_df_.groupby(['gb_variables'], as_index=False, sort=False).agg({'rank':'first', 'coeff': 'sum', 'variables':'first'})
        
        # drop hash column
        qubo_df_.drop(['gb_variables'], axis=1, inplace=True)

    return qubo_df_, key_to_qubo_dict

###############################################################################################################################

# Produce qubo matrix from qubo df

###############################################################################################################################

def qubo_matrix_from_df(qubo_mat_df):
    # populate qubo matrix dictionary
    qubo = {}
    for i in range(len(qubo_mat_df)):
        # check if index is a single var
        vars = qubo_mat_df.loc[i]['variables']
        if len(vars)==1:
            var = vars[0]
            qubo[(var, var)] = qubo_mat_df.loc[i]['coeff']
        else:
            qubo[(vars[0], vars[1])] =  qubo_mat_df.loc[i]['coeff']
    
    return qubo

# print qubo matrix
def print_qubo_matrix(qubo, key_to_qubo_dict):
    for i in range(len(key_to_qubo_dict)):
        for j in range(len(key_to_qubo_dict)):
            print(f'{qubo[(i, j)]:.1f}', end=' ') if (i,j) in qubo else print(f'%3.2f'%(0), end=' ')
        print()
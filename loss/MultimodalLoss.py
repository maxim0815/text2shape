import numpy as np
import torch

def calculate_similarity_matrix(T, S):
    '''
    calculates the dot product between T_i and S_j 
    i and j are indicies for each matrix row

    @notes:     shapes of T and S must have same size
    '''
    assert(T.shape == S.shape)
    M = torch.zeros((S.shape[0], S.shape[0]))

    for i in range(T.shape[0]):
        for j in range(S.shape[0]):
            M[i,j] = torch.dot(T[i,:], S[j,:])
    
    return M

def calculate_transition_probability(M):
    sm = torch.nn.Softmax(dim=1)
    P = sm(M)
    return P

def calculate_round_trip_loss(T, S):
    '''
    round-trip loss L_TST_R is the cross-entropy between the distribution
    P_TST and the target uniform distribution
    '''
    M = calculate_similarity_matrix(T, S)
    P_TS = calculate_transition_probability(M)
    P_ST = calculate_transition_probability(M.transpose(0,1))
    P_TST = torch.matmul(P_TS, P_ST)

    #TODO: taget uniform dist, cross-entropy

    L_TST_R = None

    return L_TST_R
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

def calculate_round_trip_loss(P_TS, P_ST):
    '''
    associating description i with certain shapes and then associating
    those shapes with description j in T is P_TST

    round-trip loss L_TST_R is the cross-entropy between the distribution
    P_TST and the target uniform distribution
    '''
    P_TST = torch.matmul(P_TS, P_ST)

    #TODO: taget uniform dist, cross-entropy

    L_TST_R = None

    return L_TST_R

def calculate_visit_loss(P_TS, m):
    '''
    loss on the probability of associating each shape with any description
    '''
    L_TST_H = None
    #TODO: 
    return L_TST_H


def cross_modal_association_loss(T, S):
    M = calculate_similarity_matrix(T, S)
    P_TS = calculate_transition_probability(M)
    P_ST = calculate_transition_probability(M.transpose(0,1))

    L_TST_R = calculate_round_trip_loss(P_TS, P_ST)
    L_TST_H = calculate_visit_loss(P_TS, m=1)

    return L_TST_R + L_TST_H
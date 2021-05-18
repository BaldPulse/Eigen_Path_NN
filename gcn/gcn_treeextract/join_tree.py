import numpy as np
'''
legacy code for tree extraction
'''
def identify_tree_edges(paths, l_thres = 0.3, u_thres = 0.8):
    '''
    returns useful edges in terms of descending contribution
    '''
    edges = []
    for i in range(paths.shape[0]):
        s_edges = []
        for j in range(paths.shape[1]):
            if paths[i,j] > u_thres:
                s_edges.append([(j,1)])
            elif paths[i,j] > l_thres:
                s_edges.append([(j,paths[i,j])])
        edges.append(s_edges)
        edges.sort(key=lambda x: (-x[1]))
    return edges


def connect_tree(edges, A):

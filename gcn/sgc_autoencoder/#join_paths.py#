import numpy as np

def identify_edges(paths, l_thres = 0.3, u_thres = 0.8):
    '''
    creates two sets of edges based on the trained matrix
    can specify lower threshhold for soft edges and upper threshold for rigid
    edges
    input:
     paths: matrix frm NN
     l_thres: lower threshold
     u_thres: upper threshold
    outputs:
     rigid_edges: set of edges with weight greater than upper threshold
     soft_edges: set of edges with weight greater than lower threshold but
     smaller than upper threshold
    '''
    rigid_edges = []
    soft_edges = []
    for i in range(paths.shape[0]):
        rigid_s_edges = []
        soft_s_edges = []
        for j in range(paths.shape[1]):
            if paths[i,j] > u_thres:
                rigid_s_edges.append([j])
            elif paths[i,j] > l_thres:
                soft_s_edges.append([j])
        rigid_edges.append(rigid_s_edges)
        soft_edges.append(soft_s_edges)
    return rigid_edges, soft_edges

def find_linked_path(paths, list_A):
    '''
    joins paths that are linked 
    inputs:
     paths: list of paths. Paths are lists of edges in the traversing order
     list_A: edge adjacency matrix in list form
    output:
     1 when link is performed
     0 when no link in performed
    '''
    for i in range(len(paths)):
        for j in range(len(paths)):
            if i != j and list_A[j][0] in list_A[paths[i][-1]]:
                list_A[i].append(list_A[j])
                list_A.pop(j)
                return 1
    return 0
    

def connect_edges(rigid_edges, soft_edges, A):
    '''
    
    '''
    list_A = []
    for i in range(A.shape[0]):
        adj = []
        for j in range(A.shape[1]):
            if A[i,j] == 1:
                adj.append(j)
        A.append(adj)
    for i in range(len(rigid_edges)):
        for j in range(len(rigid_edges[i])):
            if i != j and rigid_edges[i][j] in list_A[i]:
                
            

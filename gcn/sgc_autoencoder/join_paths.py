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
    l = len(paths)
    #print('pathlength', l)
    i = 0
    while i < l:
        for j in range(len(paths)):
            #print('ij', paths[i], paths[j])
            if i != j and (paths[j][0] in list_A[paths[i][-1]]):
                paths[i] = paths[i] + paths[j]
                #print(' ', paths[i], paths)
                paths.pop(j)
                if j < i:
                    i = i-2
                else:
                    i = i-1
                l = l-1
                break
        i = i+1
    return 0
    
def adj_to_list(A):
    list_A = []
    for i in range(A.shape[0]):
        adj = []
        for j in range(A.shape[1]):
            if A[i,j] == 1:
                adj.append(j)
        list_A.append(adj)
    return list_A

def connect_edges(rigid_edges, soft_edges, A):
    '''
    connect the available edges, favors connections edges in the same set
    input:
     rigid_edges: edges that are weighed high enough for it to be considered rigid
     soft_edges: edges that re weighed high enough to be significant but not high enough to be considered rigid
    output:
     all_paths: all paths that are found
    '''
    list_A = adj_to_list(A)
    paths = list(rigid_edges)
    for i in range(len(paths)):
        find_linked_path(paths[i], list_A)
    print('found raw connections', paths)
    #try to add edges to connect sets with only two existing paths
    '''for i in range(len(paths)):
        if len(paths[i]) == 2:
            paths[i][0].append(paths[i][1])
            paths[i].pop(1)'''
    all_paths = []
    broken_paths =[]
    for i in range(len(paths)):
        if len(paths[i]) == 1:
            all_paths.append(paths[i][0])
            continue
        for j in range(len(paths[i])):
            broken_paths.append(paths[i][j])
    ind = 1
    if broken_paths == []:
        ind = 0
    else:
        print("found broken paths", broken_paths)
        find_linked_path(broken_paths, list_A)
    all_paths+=(broken_paths)
    return all_paths, ind

#test code
if __name__ == "__main__":
    A   = np.array([[0,1,1,1,0,0,0,0,0,0],
                    [1,0,0,1,1,0,0,0,0,0],
                    [1,1,0,1,0,0,0,0,0,0],
                    [1,1,1,0,1,0,0,0,0,0],
                    [0,1,0,1,0,1,0,0,0,0],
                    [0,0,0,0,1,0,1,0,0,1],
                    [0,0,0,0,0,1,0,1,0,1],
                    [0,0,0,0,0,0,1,0,1,0],
                    [0,0,0,0,0,0,0,1,0,1],
                    [0,0,0,0,0,1,1,0,1,0]])
    list_A = []
    for i in range(A.shape[0]):
        adj = []
        for j in range(A.shape[1]):
            if A[i,j] == 1:
                adj.append(j)
        list_A.append(adj)

    p = [[5],[1],[2],[6],[7],[4]]
    print(list_A)
    find_linked_path(p, list_A)
    print(p)

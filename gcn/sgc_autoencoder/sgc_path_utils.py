import numpy as np

def binarize_paths(paths, lthres=0.1, uthres=0.9):
    bin_paths = np.array(paths)
    for i in range(paths.shape[0]):
        for j in range(paths.shape[1]):
            if paths[i,j] < lthres:
               bin_paths[i,j] = 0
            elif paths[i,j] > uthres:
                bin_paths[i,j] = 1
            else:
               bin_paths[i,j] = paths[i,j]
    return bin_paths

def weighted_jaccard(a, b):
    a_pad = a
    b_pad = b
    #pad
    if b.shape[0] > a.shape[0]:
        N = b.shape[0] - a.shape[0]
        a_pad = np.pad(a, (0,N), 'constant')
    elif a.shape[0] > b.shape[0]:
        N = a.shape[0] - b.shape[0]
        b_pad = np.pad(b, (0,N), 'constant')
    return (np.sum(a_pad) + np.sum(b_pad)- np.sum(np.absolute(a_pad-b_pad)))/(np.sum(a_pad) \
                                            + np.sum(b_pad) + np.sum(np.absolute(a_pad-b_pad)))


def jaccard_multiset(s, m):
    vm = np.zeros(s.shape[0])
    vs = np.ones(s.shape[0])
    for learned_path in m:
        max_sim = -1
        max_i = -1
        for i in range(s.shape[0]):
            sim = weighted_jaccard(learned_path, s[i])
            if sim>max_sim:
                max_sim = sim
                max_i = i
        vm[max_i] += max_sim
    print(vm, vs)
    return weighted_jaccard(vm, vs), vm, vs


def evaluate_path_similarities(epath, lpath):
    return jaccard_multiset(epath, lpath)

def l1_normalize(A):
    return A / np.sum(np.absolute(A), axis=1).reshape(A.shape[0],1)

if __name__ == "__main__":
    t0 = np.array([[1,2,3,1],
                   [2,0,0,1]])
    t0 = t0 / np.sum(np.absolute(t0), axis=1).reshape(t0.shape[0],1)
    t1 = np.array([[1,2,3,0],
                   [1,0,0,1]])
    t1 = t1 / np.sum(np.absolute(t1), axis=1).reshape(t1.shape[0],1)
    #print(t0, t0,weighted_jaccard(t0, t0))
    #print(t0, t1,weighted_jaccard(t0, t1))
    print(t0, t1)
    print(jaccard_multiset(t0, t1))

    p0 = np.array([[0.06,1, 0.2, 0.023],
                   [0.66666667, 0, 0, 0.33333333]])
    print(p0, '\n', binarize_paths(p0))
    #print(weighted_jaccard([1,2,3,1], [1,2,3,0]))
    #print(weighted_jaccard([1,2,3,1], [2,0,0,1]))
    #print(weighted_jaccard([1,0,0,1], [2,0,0,1]))

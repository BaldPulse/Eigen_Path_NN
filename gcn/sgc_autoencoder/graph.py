import tensorflow as tf
import numpy as np

class Network:
    node_adj = None
    edge_adj = None
    nNodes = 0
    Edges = None
    nEdges = 0

    def __init__(self, adj):
        self.node_adj = adj
        self.nNodes = adj.shape[0]
    def adj_to_edge_adj(self, adj):
        '''
        transforms node adjacency matrix to edge adjacency matrix
        '''
        nEdges = 0
        Edges = []
        if(adj is None):
            adj = self.node_adj
        for start in range(adj.shape[0]):
            for dest in range(adj.shape[0]):
                if(adj[start, dest] == 1):
                    nEdges += 1
                    Edges.append([start, dest])
        
        edge_adj = np.zeros((nEdges, nEdges), dtype = int)
        # for each edge in Edges[], check their connection to other edges via the adjacency matrix. 
        # if they are connected, set the entry in edge_adj to 1
        for e in range(nEdges):
            for i in range(adj.shape[0]):
                if(adj[Edges[e][1], i] == 1):
                    edge_adj[e, Edges.index([Edges[e][1], i])] = 1
        
        self.nEdges = nEdges
        self.Edges = Edges
        self.edge_adj = edge_adj
        return edge_adj

    def path_to_node_prop(self, path):
        '''
        transforms path array to a node propagation matrix
        '''
        node_prop = np.zeros((self.nNodes, self.nNodes))
        for e in range(len(path)):
            node_prop[self.Edges[e][1], self.Edges[e][0]] = path[e]
            # node_prop[self.Edges[e][0], self.Edges[e][0]] = 1 - path[e]
        node_prop[0, :] = 1-np.sum(node_prop, axis=0)
        return node_prop

class Network_flows:
    '''
    represents flow snapshot data in terms of nodes and edges
    '''
    def __init__(self, flows, network):
        self.edge_flows = flows
        self.netWork = network

    def edge_flows_to_node_flows(self):
        self.node_flows = []
        for f in self.edge_flows:
            node_flow = [0] * self.netWork.nNodes
            for e in range(self.netWork.nEdges):
                node_flow[self.netWork.Edges[e][0]] -= f[e]
                node_flow[self.netWork.Edges[e][1]] += f[e]
            self.node_flows.append(node_flow)



class Propagation:
    def __init__(self, netWork, paths, sources, sinks):
        self.netWork = netWork
        self.node_props = []
        self.sources = sources
        self.sinks = sinks
        

    def propagate(self,factors, paths):
        '''
        factor: # of flows*p array factor for each path, wich will be used as the starting flow in each source node
        note: path and factor are aligned such that paths*factor will give the L2 loss
        '''
        prop_flows = np.zeros((factors.shape[0], self.netWork.nNodes))
        flow_count = 0
        node_prop  = []
        for p in paths:
            mult = self.netWork.path_to_node_prop(p)
            node_prop.append(mult)
        print("node prop \n", node_prop)
        for F in factors:
            prop_flow = np.zeros((F.shape[0], self.netWork.nNodes)) # # of paths * # of edges
            for i in range(len(F)):
                for s in self.sources:
                    prop_flow[i, s] = F[i] #set every source to the path factor
                iter = np.count_nonzero(paths[i])
                for j in range(iter):
                    prop_flow[i] = np.dot(node_prop[i], prop_flow[i].T).T
                for s in self.sources:
                    prop_flow[i, s] -= F[i] #reset source by path factor to account for outgoing flows
            prop_flows[flow_count] = np.sum(prop_flow, axis=0)
            flow_count = flow_count+1
        return prop_flows
                

if __name__ == "__main__":
#test code
    example_node_adj = np.array([[0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1],
                                [0, 1, 0, 1, 1],
                                [1, 0, 1, 0, 0],
                                [0, 1, 1, 0, 0]])

    example_edge_adj = np.array([[0,0,1,1,1,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,1,1,0,0],
                                [1,1,0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,1,1,1,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0,1,1],
                                [0,0,1,1,1,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,1,1,0,0],
                                [0,0,0,0,0,0,0,0,0,0,1,1],
                                [1,1,0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,1,1,1,0,0,0,0],
                                [0,0,1,1,1,0,0,0,0,0,0,0],
                                [0,0,0,0,0,1,1,1,0,0,0,0]])

    example_edge_flow = np.array([[1,0,0,0,1,0,0,0,0,0,0,0],
                                [0,1,0,0,0,0,0,1,0,1,0,0]],
                                float)

    example_factors = np.array([[1,0],
                                [0,1]],
                                float)

    example_paths = np.array([[1,0,0,0,1,0,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0,1,0,1,0,0]])

    example_soft_paths = np.array([[1,0,0,0,0.5,0,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0,1,0,0.5,0,0]])

    example_network = Network(example_node_adj)
    test_edge_adj = example_network.adj_to_edge_adj(example_node_adj)
    # print(np.array_equiv(test_edge_adj, example_edge_adj), (test_edge_adj-example_edge_adj))
    # print(example_edge_adj)

    example_flow = Network_flows(example_edge_flow, example_network)
    example_flow.edge_flows_to_node_flows()
    print(example_flow.node_flows)

    example_propagation = Propagation(example_network, example_paths, [0], [4])
    prop_result = example_propagation.propagate(example_factors, example_paths)
    print(prop_result)

    prop_result = example_propagation.propagate(example_factors, example_soft_paths)
    print(prop_result)

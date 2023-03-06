"""
Author: V Vijendran
Email: v.vijendran@anu.edu.au
"""


import sys
import itertools
import numpy as np
import networkx as nx
from scipy import optimize
from functools import partial
from numpy import sin, cos, pi
import matplotlib.pyplot as plt


def draw_graph(G):
    """
        Given a weighted NetworkX Graph 'G', this function will plot the graph.

        This  "draw_graph" function is written for the purpose the debugging 
        RQAOA after each step of variable elimination.
    """

    # Create a Matplotlib Figure and place the graph vertices in a circular layout
    plt.figure(figsize=(10,10))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=250)
    
    # Get all the edges with postive weights.
    epositive = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.0]
    # Get all the edges with negative weights.
    enegative = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < 0.0]
    
    # Draw the positive and negatively weighted edges each with their own defined style.
    nx.draw_networkx_edges(G, pos, edgelist=epositive, width=3)
    nx.draw_networkx_edges(G, pos, edgelist=enegative, width=3, alpha=0.5, edge_color="b", style="dashed")


    # Draw the node labels on the figure.
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    # Get the edge weights and draw them on the figure.
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15);


def has_edge(edge, edge_list):
    """
        Given an edge '(u, v)' and a list of edges, 'edge_list',
        the 'has_edge' function simply checks if the the edge
        '(u, v)' or '(v, u)' is present in the list. 

        In an undirected graph, '(u, v)' is the same as '(v, u)'.
    """
    return (edge in edge_list) or (edge[::-1] in edge_list)


def QAOA_Expectation(graph, angles, optim=True):
    """
        Given a weighted NetworkX Graph 'graph' and angles = [gamma, beta],
        this function computes the expectation value of the level-1 QAOA
        for the MaxCut problem.

        The formula used for computing the expectation value for level-1
        QAOA on MaxCut is based on Eq 13 from https://arxiv.org/abs/2302.04479.

        The third argument 'optim' is a boolean variable used to indicate
        whether or not an optimiser is calling the function. If the optimiser
        is calls this function, then the cost for angles = [gamma, beta] is
        returned. Otherwise a dictionary containing the mapping b/w the edges
        and their corresponding costs is returned. This dictionary is used for
        the variable elimination step of the RQAOA.
    """

    # This dictionary cotains the edges and their assosicated expectation values.
    edge_costs = {}
    # Gamma and Beta angles that were passed in as an argument.
    gamma, beta = angles

    # Iterate through all edges (u, v) of the given graph.
    for u, v in graph.edges():
        
        # Get the weight of the edge (u, v)
        w_uv = graph[u][v]['weight']
        # e is the set of vertices other than u that are connected to v.
        e = [w for w in graph[v]]
        e.remove(u)
        # d is the set of vertices other than v that are connected to u.
        d = [w for w in graph[u]]
        d.remove(v)
        # F is the set of vertices that form a triangle with the edge (u, v).
        # In other words, F is the set of vertices that are neighbours of both
        # u and v.
        F = list(set(e).intersection(d))

        # Get all the edges that are connected to either 'u' or 'v'.
        S = [(x, v) for x in e] + [(u, y) for y in d]
        S = list(set(S)) # We use list(set()) to remove duplicates if any.
        
        # Compute the first term in the square brackets of Eq 13.
        term1_cos1 = np.prod([cos(gamma * graph[x][v]['weight']) for x in e])
        term1_cos2 = np.prod([cos(gamma * graph[u][y]['weight']) for y in d])
        term1 = sin(4 * beta) * sin(gamma * w_uv) * (term1_cos1 + term1_cos2)
    
        # E is the list of all edges that are connected to either 'u' or 'v' 
        # but not both. In other words, E is the list edges that do not form
        # a triangle with the edge (u, v).
        e_edges_non_triangle = [(x, v) for x in e if x not in F]
        d_edges_non_triangle = [(u, y) for y in d if y not in F]
        E =  e_edges_non_triangle + d_edges_non_triangle
        E = list(set(E))

        # Compute the second term in the square brackets of Eq 13.
        term2 = pow(sin(2 * beta), 2) * np.prod([cos(gamma * graph[x][y]['weight']) for x, y in E])
        triangle_1_terms = np.prod([cos(gamma * (graph[u][f]['weight']  + graph[v][f]['weight'])) for f in F])
        triangle_2_terms = np.prod([cos(gamma * (graph[u][f]['weight']  - graph[v][f]['weight'])) for f in F])
        term2 = term2 * (triangle_1_terms - triangle_2_terms)

        # ZuZv is simply the <Z_u Z_v> value.
        ZuZv = -0.5 * (term1 + term2)

        # If an optimiser is calling this function the edge_cost is the MaxCut cost;
        # otherwise, it is simply the <Z_u Z_v> value.
        if optim:
            edge_costs[(u, v)] = (w_uv/2) * (1 - ZuZv)
        else:
            edge_costs[(u, v)] = ZuZv
            
    # If an optimiser is calling this function simple return the total MaxCut cost;
    # otherwise, return the dictionary containing the edges and their correlation value.
    if optim:
        total_cost = sum(edge_costs.values())
        return -total_cost
    else:
        return edge_costs


class GraphManager:
    """
        The GraphManager class is used by the RQAOA method. The purpose of the
        GraphManager class is three-fold:

            1. It stores the original input graph that is used for computing the
               the cost at the end of the RQAOA process.

            2. The GraphManager is used in the variable elimination step of the
               RQAOA process. Depending on the magnitude and the sign of the 
               correlation, the variable is removed and it's mapping is stored in
               the 'node_maps' dictionary. The corresponding edge is removed and 
               reattached to the mapped vertex. This is the most important step
               as it changes the problem, and we need to carry out this change
               error free.

            3. The GraphManager contains a dictionary 'node_maps' that maps the 
               eliminated nodes to other eliminated or non-eliminated nodes. This mapping
               is important to find the optimal solution to the original graph.
    """
    
    def __init__(self, graph, verbose = False):
        """
            The GraphManager class needs only two values to be initialized; a
            weighted NetworkX Graph, and a boolean variable 'verbose' indicating
            whether the GraphManager should print every activity.
        """
        # This variable stores the original graph. No changes are made to this variable.
        self.original_graph = graph.copy() 
        # This variable stores the graph after each variable elimination steps of the RQAOA.
        self.reduced_graph = graph
        # Boolean value to indicate whether or not to print activity.
        self.verbose = verbose
        # This dictionary maps each node to a value in {1, -1}. This dictionary is used
        # to store the optimal assignments for the original problem.
        self.nodes_vals = {i : 0 for i in range(graph.number_of_nodes())}
        # This dictionary contains the mapping from one node to another based upon the
        # RQAOA's variable elimination method. Initially all nodes are mapped to themselves
        # with a +1 correlation. 
        self.node_maps = {i : (i, 1) for i in range(graph.number_of_nodes())}
        # This list contains all the nodes that aren't eliminated yet. When the GraphManager
        # is first initialized, this list contains all nodes.
        self.remaining_nodes = [i for i in range(graph.number_of_nodes())]

    def correlate(self, edge):
        """
            Given an edge (u, v) this function maps u = v and removes the node u from the
            graph. Any edges that are connected to 'u' are removed and are added to vertex
            'v'.        
        """
        # Get the vertices u and v.
        u, v = edge
        # Make sure the reduced graph has the edge (u, v) or else you are trying to remove
        # something that is not there.
        assert self.reduced_graph.has_edge(u, v), f"Graph does not contain edge ({u},{v})."
        # d is the set of vertices other than v that are connected to u.
        d = [w for w in self.reduced_graph[u]]
        d.remove(v)
        # Anti-Correlate by setting the map of u is v; i.e. we are eliminating the node u.
        self.node_maps[u] = (v, 1)
        self.remaining_nodes.remove(u)
        if self.verbose:
            print(f"Removing edge ({v}, {u}) with weight {self.reduced_graph[v][u]['weight']} from graph.")
        # Remove the edge (u, v) from the reduced graph.
        self.reduced_graph.remove_edge(v, u)
        # Get the weights of all the edges connected to the vertex 'u'.
        old_weights = {w : self.reduced_graph[w][u]['weight'] for w in d}
        
        # Iterate through all the neighbours of the vertex 'u' and remove their edges from the
        # reduced_graph.
        for w in d:
            if self.verbose:
                print(f"Removing edge ({w}, {u}) with weight {self.reduced_graph[w][u]['weight']} from graph.")
            self.reduced_graph.remove_edge(w, u)
        # Remove the vertex 'u' from the reduced_graph.
        if self.verbose:
            print(f"Removing node {u} from graph.")
        self.reduced_graph.remove_node(u)
        
        # Make new edges connecting the neighbours of the vertex 'u' to the vertex 'v'.
        # Use the old weights for the time being.
        new_edges = {(w, v) : old_weights[w] for w in d}
        # Iterate thorugh all the new edges.
        for new_edge in new_edges:
            # If the reduced_graph already has the new_edge, then simply update the weights by 
            # summing the existing_weight with the weight of the edge that was previously removed.
            if self.reduced_graph.has_edge(new_edge[0], new_edge[1]):
                new_edges[new_edge] += self.reduced_graph[new_edge[0]][new_edge[1]]['weight']

        # Iterate through all the edges (w, v) and weight from the 'new_edges' dictionary.
        for new_edge, weight in new_edges.items():
            if weight == 0.0:
                # If there are any edges with zero weight (that may have occurred to the previous loop),
                # then remove that edge from the reduced_graph.
                if self.verbose:
                    print(f"Removing edge {new_edge} with weight {weight} from graph.")
                self.reduced_graph.remove_edge(new_edge[0], new_edge[1])
            else:
                # Add all the edges with the non-zero weight to the reduced_graph datastructure.
                if self.verbose:
                    print(f"Adding edge {new_edge} with weight {weight} to graph.")
                self.reduced_graph.add_edge(new_edge[0], new_edge[1], weight = weight)
        

    def anti_correlate(self, edge):
        """
            Given an edge (u, v) this function maps u = -v and removes the node u from the
            graph. Any edges that are connected to 'u' are removed and are added to vertex
            'v'.

            This is the exact same as the 'correlate' function except there is -ve sign at
            certain locations of the code due to the -1 correlation b/w the vertices u and v.
        """
        u, v = edge
        assert self.reduced_graph.has_edge(u, v), f"Graph does not contain edge ({u},{v})."
        # d is the set of vertices other than v that are connected to u.
        d = [w for w in self.reduced_graph[u]]
        d.remove(v)
        # Anti-Correlate by setting the map of u is v; i.e. we are eliminating the node u.
        self.node_maps[u] = (v, -1)
        self.remaining_nodes.remove(u)
        if self.verbose:
            print(f"Removing edge ({v}, {u}) with weight {self.reduced_graph[v][u]['weight']} from graph.")
        self.reduced_graph.remove_edge(v, u)
        
        old_weights = {w:self.reduced_graph[w][u]['weight'] for w in d}
        for w in d:
            if self.verbose:
                print(f"Removing edge ({w}, {u}) with weight {self.reduced_graph[w][u]['weight']} from graph.")
            self.reduced_graph.remove_edge(w, u)
        if self.verbose:
            print(f"Removing node {u} from graph.")
        self.reduced_graph.remove_node(u)
        new_edges = {(w, v):-old_weights[w] for w in d}

        for new_edge in new_edges:
            if self.reduced_graph.has_edge(new_edge[0], new_edge[1]):
                new_edges[new_edge] += self.reduced_graph[new_edge[0]][new_edge[1]]['weight']

        for new_edge, weight in new_edges.items():
            if weight == 0.0:
                if self.verbose:
                    print(f"Removing edge {new_edge} with weight {weight} from graph.")
                self.reduced_graph.remove_edge(new_edge[0], new_edge[1])
            else:
                if self.verbose:
                    print(f"Adding edge {new_edge} with weight {weight} to graph.")
                self.reduced_graph.add_edge(new_edge[0], new_edge[1], weight = weight)

            
    def get_root_node(self, node, s):
        """
            This is a recursive funtion that takes an eliminated node and + or - correlation sign 's'
            and returns the root node with a + or - correlation sign. A root node is a node that has 
            not been eliminated in the final step of RQAOA.

            The reason for this recursive function is because not all eliminated nodes are mapped to 
            a root node. They might be mapped to other eliminated node which maps to other eliminated 
            node and so on until an eliminated node is mapped to root node. All these mappings have
            a +ve or -ve mapping. With the use of this recursive function we can trace the path of the
            mappings all the way to the root node and also correctly keep track of the sign changes. 
        """
        # For a given 'node', 'mapped_tuple' is simply the tuple that contains new node 
        # that 'node' is mapped to along with the + or - sign.
        mapped_tuple = self.node_maps[node]
        mapped_node, sign = mapped_tuple # Unpacking the tuple. 
        sign = sign * s # Update the sign based the argument 's'
        # If the 'mapped_node' is indeed the root node then return it along with the sign,
        # else, recurse by calling get_root_node(mapped_node, sign).
        if mapped_node in self.remaining_nodes:
            return mapped_node, sign
        else:
            return self.get_root_node(mapped_node, sign)
            
    def set_node_values(self, values):
        """
            Given a list 'values' containing +1 or -1 (where len(values) == len(remaining_nodes)), this function assigns those
            integer values to the remaining nodes, and propagates the values to the eliminated nodes by using the 'node_maps'
            dictionary.
        """
        assert len(values) == len(self.remaining_nodes), "Number of values passed is not equal to the number of remaining nodes."
        for value in values:
            assert value == 1 or value == -1, "Values passed should be either 1 or -1."
            
        # Set the values of remaining set of nodes; i.e. the nodes that have been eliminated.
        for i, value in enumerate(values):
            node = self.remaining_nodes[i]
            self.nodes_vals[node] = value
            
        # Propagate the values of eliminated nodes based on the node mappings.
        for node, mapped_tuple in self.node_maps.items():
            mapped_node, sign = mapped_tuple
            if node != mapped_node: # Skip Root Nodes (Only root nodes are mapped to themselves.)
                if mapped_node in self.remaining_nodes: #If map leads to a root node, apply the value.
                # The minus sign indicates that the two nodes are anti-correlated.
                    self.nodes_vals[node] = sign * self.nodes_vals[mapped_node]
                else:
                    root_node, s = self.get_root_node(mapped_node, sign)
                    self.nodes_vals[node] = s * self.nodes_vals[root_node]
                     
                
    def compute_cost(self, graph):
        """
            The 'compute_cost' function simply computes the MaxCut cost for the graph.
            The 'graph' argument can be used to specify whether we would like to compute the MaxCut cost
            for the 'original_graph' or the 'reduced_graph'.
        """
        # This for loop simply checks if the assignent values are either +1 or -1.
        for value in  self.nodes_vals.values():
            assert value == 1 or value == -1, "All nodes should have a value of either 1 or -1."

        total_cost = 0
        for edge in graph.edges(): 
                total_cost += 0.5*(graph[edge[0]][edge[1]]['weight'])*(1 - self.nodes_vals[edge[0]]*self.nodes_vals[edge[1]])
    
        return total_cost  
    
    
    def brute_force(self, minimise):
        """
            The 'brute_force' function simply iterates through all possible assignment of +1 and -1
            for the 'reduced_graph' and chooses the assignment that maximises or minimises the objective
            function for the 'reduced_graph'. The optimal assignments for the 'reduced_graph' will also
            be the optimal assignments for the 'original_graph'.
        """
        
        num_values = len(self.remaining_nodes)
        assignments = list(map(list, itertools.product([1, -1], repeat=num_values)))
        
        if minimise:
            best_reduced_cost = 10000
        else:
            best_reduced_cost = -10000

        best_assignment = assignments[0]

        
        for i, assignment in enumerate(assignments):
            self.set_node_values(assignment)
            reduced_cost = self.compute_cost(self.reduced_graph)
            if minimise:
                if reduced_cost < best_reduced_cost:
                    best_reduced_cost = reduced_cost
                    best_assignment = assignment
                    print(f"Best reduced cost found so far is {best_reduced_cost} for assignment {assignment}.")
            else:
                if reduced_cost > best_reduced_cost:
                    best_reduced_cost = reduced_cost
                    best_assignment = assignment
                    print(f"Best reduced cost found so far is {best_reduced_cost} for assignment {assignment}.")

        self.set_node_values(best_assignment)
        best_cost = self.compute_cost(self.original_graph)

        print(f"Best Cost found for the original problem is {best_cost}.")

        return best_cost, self.nodes_vals

def eliminate_variable(graphmanager: GraphManager):
        """
            The 'eliminate_variable' function is the vital function that performs variable elimination step in 
            the RQAOA process. 
        """
        
        # We make a partial function 'qaoa' which has only one input argument which is 'angles = [gamma, beta]'
        # so it is more compatible with the optimiser.
        qaoa = partial(QAOA_Expectation, graphmanager.reduced_graph)
        # We use a brute-force search to optimise each step of RQAOA. The 'workers' argument allows to run the
        # brute-force search in parallel. Ideally set 'workers' to equal to number of virtual cpus.
        res = optimize.brute(qaoa, ((0, 2*pi), (0, 2*pi)), workers = 16)
        qaoa_cost = -qaoa(res) # Get the QAOA cost.
        # Get the correlation b/w each edge and the sort the edges with in descending order of the magnitude of correlation.
        edge_costs = QAOA_Expectation(graphmanager.reduced_graph, res, optim = False) 
        edge_costs = {k: v for k, v in sorted(edge_costs.items(), key=lambda item: np.abs(item[1]), reverse = True)}
        edge, weight = list(edge_costs.items())[0] # Get the edge with highest correlation
        sign = int(np.sign(sys.float_info.epsilon+weight)) # Get the sign of the correlation; this is either +1 or -1.

        # Either correlate or anti-correlate depending on the sign.
        if sign < 0:
            if graphmanager.verbose:
                print(f"QAOA Cost = {qaoa_cost}. Anti-Correlating Edge {edge} that has maximum absolute weight {weight}")
            graphmanager.anti_correlate(edge)
        elif sign > 0:
            if graphmanager.verbose:
                print(f"QAOA Cost = {qaoa_cost}. Correlating Edge {edge} that has maximum absolute weight {weight}")
            graphmanager.correlate(edge)
        else:
            print(f"Cannot correlate or anti-correlate edge {edge} for weight {weight}.")

def RQAOA(graphmanager: GraphManager, n, minimise = False):
    # Perform Variable Elimation until i == n.
    i = 0
    while i <= n:
        print(f"Iter {i}: Graph has {graphmanager.reduced_graph.number_of_nodes()} nodes and {graphmanager.reduced_graph.number_of_edges()} edges remaining.")
        eliminate_variable(graphmanager)
        i += 1

    # After eliminating 'n' variable brute-force the reduced problem.
    print("")
    print("Brute-Forcing")
    return graphmanager.brute_force(minimise)
        
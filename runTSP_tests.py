import networkx as nx
import tsplib95 as tsp
import numpy as np
from queue import PriorityQueue
import tracemalloc
import time
import numpy as np

# Implements the Branch and Bound algorithm

class Node:
    def __init__(self, bound, path, edges, cost):
        self.bound = bound
        self.path = path
        self.edges = edges
        self.cost = cost
    
    def __lt__(self, other):
        if len(self.path) == len(other.path): #longer paths mean we are closer to the complete solution
            return self.cost < other.cost #prioritize lower costs
        else:
            return len(self.path) > len(other.path)
    
def findSmallestEdges(problem, node):
    min1 = float('inf')
    min2 = float('inf')
    for other_node in problem.get_nodes():
        if other_node != node:
            weight = problem.get_weight(node, other_node)
            if weight < min1:
                min2 = min1
                min1 = weight
            elif weight < min2:
                min2 = weight
    return min1, min2

def computeFirstBound(problem):
    bound = 0
    num_nodes = problem.dimension
    initialBoundEdges = np.zeros((num_nodes + 1, 2))

    for i in range(1, num_nodes + 1):
        min1, min2 = findSmallestEdges(problem, i)
        initialBoundEdges[i][0] = min1
        initialBoundEdges[i][1] = min2
        bound += min1 + min2

    return bound / 2, initialBoundEdges

def computeNodeBound(problem, path, edges, bound):
    num_nodes = problem.dimension + 1
    changedEdges = np.zeros(num_nodes, dtype=int)
    newEdges = np.array(edges)

    # Get the weight of the most recently added edge in the path
    edgeWeight = problem.get_weight(path[-2], path[-1])
    
    # Double the current bound to adjust for calculations
    sum = bound * 2

    # Update for the second-to-last node in the path
    if newEdges[path[-2]][0] != edgeWeight:
        if changedEdges[path[-2]] == 0:
            sum -= newEdges[path[-2]][1]
            sum += edgeWeight
        else:
            sum -= newEdges[path[-2]][0]
            sum += edgeWeight
        changedEdges[path[-2]] += 1

    # Update for the last node in the path
    if newEdges[path[-1]][0] != edgeWeight:
        if changedEdges[path[-1]] == 0:
            sum -= newEdges[path[-1]][1]
            sum += edgeWeight
        else:
            sum -= newEdges[path[-1]][0]
            sum += edgeWeight
        changedEdges[path[-1]] += 1

    # Return the updated bound and the updated edges array
    return sum / 2, newEdges

def branchAndBoundTSP(problem):
    tracemalloc.start()
    initial_time = time.time()
    initialBound, initialBoundEdges = computeFirstBound(problem)
    root = Node(initialBound, [1], initialBoundEdges, 0)
    pq = PriorityQueue()
    pq.put(root)
    best = float('inf')
    best_solution = []
    num_nodes = problem.dimension

    while not pq.empty():
        node = pq.get()
        level = len(node.path)

        # Check if a complete solution is found
        if level == num_nodes:
            if node.cost < best:
                best = node.cost
                best_solution = node.path 
        else:
            # Explore further only if the current bound is less than the best known cost
            if node.bound < best:
                for k in range(1, num_nodes + 1):
                    if k not in node.path:
                        new_solution = node.path + [k]
                        edgeWeight = problem.get_weight(node.path[-1], k)
                        newBound, newEdges = computeNodeBound(problem, new_solution, node.edges, node.bound)

                        # Check if the new node is promising
                        if newBound < best:
                            new_cost = node.cost + edgeWeight
                            newNode = Node(newBound, new_solution, newEdges, new_cost)
                            pq.put(newNode)

    best = best + problem.get_weight(best_solution[-1], 1) #add the weight of the last edge to the first node
    best_solution.append(1)

    final_time = time.time()
    total_time = final_time - initial_time
    memory, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return best, best_solution, total_time, memory

# Implements Twice-Around-the-Tree algorithm

def findPathWeight(problem, path):
    weight = 0
    for i in range(len(path) - 1):
        weight += problem.get_weight(path[i], path[i + 1])
    return weight

def twiceAroundTheTreeTSP(problem):
    tracemalloc.start()
    initial_time = time.time()
    A = problem.get_graph()
    MST = nx.minimum_spanning_tree(A)
    path = list(nx.dfs_preorder_nodes(MST, 1))  # Assuming the nodes start from 1
    path.append(path[0])
    weight = findPathWeight(problem, path)
    final_time = time.time()
    total_time = final_time - initial_time
    memory, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return weight, total_time, memory


# Implements Christofides algorithm

def findShortcutPath(graph):
    path = list(nx.eulerian_circuit(graph, 1))
    path = [x[0] for x in path]

    # Remove duplicates
    shortcutPath = list(dict.fromkeys(path))
    
    return shortcutPath + [shortcutPath[0]]

def christofidesTSP(problem):
    tracemalloc.start()
    initial_time = time.time()
    graph = problem.get_graph()

    MST = nx.minimum_spanning_tree(graph)
    degrees = nx.degree(MST)
    odd_nodes = [node for node, degree in degrees if degree % 2 == 1]
    odd_nodes_subgraph = graph.subgraph(odd_nodes)
    matching = list(nx.min_weight_matching(odd_nodes_subgraph, maxcardinality=True))    

    MST_multi_graph = nx.MultiGraph(MST)
    for node1, node2 in matching:
        MST_multi_graph.add_edge(node1, node2, weight=problem.get_weight(node1, node2))

    path = findShortcutPath(MST_multi_graph)
    weight = findPathWeight(problem, path)

    final_time = time.time()
    total_time = final_time - initial_time
    memory, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return weight, total_time, memory



# Script to run all instances

instancias = "lib/"

instancias_info = {}

with open('tp2_datasets.txt', 'r') as file:
    next(file)

    for line in file:
        dataset, nos, limiar_str = line.strip().split()
    
        if '[' in limiar_str:
            limiar = eval(limiar_str)
            limiar = sum(limiar) / len(limiar)
        else:
            limiar = int(limiar_str)

        instancias_info[dataset] = [int(nos), limiar]



# Create csv file to store results
with open('results.csv', 'w') as file:
    file.write("Instância,Algoritmo,Nós,Limiar,Resultado,Qualidade,Tempo de execução(s),Memória(bytes)\n")



twice = True
christofides = True

for arquivo in instancias_info.keys():

    
    instancia = arquivo.split(".")[0]
    file_name = arquivo + ".tsp"
    problem = tsp.load(instancias + file_name)
    print("Working on", instancia)

    nos = instancias_info[instancia][0]
    limiar = instancias_info[instancia][1]

    if twice:
        peso, tempo, memoria = twiceAroundTheTreeTSP(problem)

        qualidade = (peso / limiar)

        if tempo > 1800:
            print(" Time exceeded for ",nos, " nodes. Twice Around the Tree stopped here.")
            twice = False

        if twice:
            with open('results.csv', 'a') as file:
                file.write(f"{instancia},Twice Around the Tree,{nos},{limiar},{peso},{qualidade},{tempo},{memoria}\n")


    if christofides:
        peso, tempo, memoria = christofidesTSP(problem)
        qualidade = (peso / limiar)

        if tempo > 1800:
            print(" Time exceeded for ",nos, " node Christofides stopped here.")
            christofides = False
            
        if christofides:
            with open('results.csv', 'a') as file:
                file.write(f"{instancia},Christofides,{nos},{limiar},{peso},{qualidade},{tempo},{memoria}\n")

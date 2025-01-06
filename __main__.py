from cluster_manager import Cluster
from basis import Basis
from optimizer import Optimizer
import matplotlib.pyplot as plt

def main():
    cluster = Cluster(graph_type="Grid", num_nodes=10, squeezing_value=10, normalization_type=1)
    cluster.initialize_cluster()
    print(cluster.get_adjacency_matrix())
    # print(cluster.get_sympeigvals_alice())
    # print(cluster.is_routable())
    # cluster.show_graph()
    
    # Building the basis for optimization
    basis = Basis(cluster)
    basis.initialize_basis()
    
    # Création et exécution de l'optimiseur
    optimizer = Optimizer(cluster, basis)
    optimizer.initialize_optimizer(generation_number=5001,objective_function="standard", alice_node_number=0, bob_node_number=cluster.num_nodes//2)
    optimizer.optimize()
    optimizer.show_solution()
    
if __name__ == "__main__":
    main()

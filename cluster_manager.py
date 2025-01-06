import networkx as nx
import numpy as np
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson
import strawberryfields as sf
from scipy.linalg import fractional_matrix_power
from scipy.linalg import LinAlgError
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt


class Cluster:
    def __init__(self, graph_type="BA", num_nodes=50, squeezing_value=10, normalization_type=1):
        """
        Initialize the graph manager with a specified type and number of nodes.
        """
        self.graph_type = graph_type
        self.num_nodes = num_nodes
        self.squeezing_value = squeezing_value
        self.normalization_type = normalization_type
        
        
    def initialize_cluster(self):
        self.generate_graph()
        # self.compute_distance_matrix()
        self.build_permutation_matrix(self.num_nodes)
        self.initialize_covariance_matrix(self.adjacency_matrix, self.squeezing_value, self.normalization_type)
        self.covariance_matrix_convention_paper = self.permutation_matrix @ self.covariance_matrix @ self.permutation_matrix.T
        self.alice_cov, self.bob_cov = self.covariance_matrix_convention_paper[:self.num_nodes,:self.num_nodes], self.covariance_matrix_convention_paper[self.num_nodes:,self.num_nodes:]
        self.symp_eig_alice, self.Williamson_alice, self.BM_alice = self.compute_eigenvalues(self.alice_cov)
        self.symp_eig_bob, self.Williamson_bob, self.BM_bob = self.compute_eigenvalues(self.bob_cov)
        
        self.lambda_val = np.round(self.squeezing_value/2 + 1/(2*self.squeezing_value),4)
        self.mu_val = np.round(self.squeezing_value/2 - 1/(2*self.squeezing_value),4)
        
    def generate_graph(self):
        """
        Generate the graph based on the specified type.
        """
        if self.graph_type == "ER":
            self.graph = nx.erdos_renyi_graph(self.num_nodes, 0.5)
        elif self.graph_type == "BA":
            self.graph = nx.barabasi_albert_graph(self.num_nodes, 2)
        elif self.graph_type == "WS":
            self.graph = nx.watts_strogatz_graph(self.num_nodes, 4, 0.1)
        elif self.graph_type == "Grid":
            self.graph = nx.grid_graph(dim=[self.num_nodes // 2, 2])
        elif self.graph_type == "Full":
            self.graph = nx.complete_graph(self.num_nodes)
        else:
            raise ValueError("Unknown graph type")
        self.adjacency_matrix = nx.to_numpy_array(self.graph)
        
    def build_permutation_matrix(self, num_nodes):
        m = np.eye(2*num_nodes)
        for i in range(num_nodes//2,num_nodes):
            a = m[i,:].copy()
            b = m[i+num_nodes//2,:].copy()
            m[i,:],m[i+num_nodes//2,:] = b,a
        self.permutation_matrix = m
        
    def initialize_covariance_matrix(self, adjacency_matrix, squeezing_value, normalization_type):
        n = len(adjacency_matrix)
        I = np.eye(n)
        O = I # Set O to the identity matrix for now
        X = fractional_matrix_power(I + matrix_power(adjacency_matrix, 2), -1 / 2).dot(O)
        Y = adjacency_matrix.dot(fractional_matrix_power(I + matrix_power(adjacency_matrix, 2), -1 / 2).dot(O))
        S_A = np.block([[X, Y], [-Y, X]]) # Symplectic matrix
        
        L = np.array([squeezing_value for i in range(n)])
        pvar = np.array(np.power(10, L / 10) * np.power(normalization_type, 2))
        xvar = 1 / pvar
        S = np.diag(np.concatenate((xvar, pvar))) # Squeezing matrix
        self.covariance_matrix = S_A.dot(S.dot(np.transpose(S_A))).real
        
    def compute_eigenvalues(self, covariance_matrix):
        """
        Compute the eigenvalues of the covariance matrix.
        """
        A = np.round(covariance_matrix,8)
        wa,SA = sf.decompositions.williamson(A)
        wa = np.round(np.diagonal(wa),4)
        SA = np.linalg.inv(SA)
        UA = sf.decompositions.bloch_messiah(SA)[2]
        return wa,SA,UA
    
    def is_routable(self, internal_routing=False):
        """
        Check if the graph is routable between Alice and Bob based on the symplectic eigenvalues.
        """
        if internal_routing:
            return np.count_nonzero(self.symp_eig_alice == 1) >= 2
        else:
            print(self.lambda_val)
            return np.count_nonzero(self.symp_eig_alice == self.lambda_val) >= 1
        
    def compute_distance_matrix(self):
        "Renvoie la matrice du plus cours chemin entre chaque paire de noeud du graph G"
        dist_mat = np.empty((self.num_nodes,self.num_nodes))
        dpath = dict(nx.all_pairs_shortest_path_length(self.graph))
        for i in range(len(dist_mat)) :
            for j in range(len(dist_mat)) :
                dist_mat[i][j] = dpath[i][j]
        self.distance_matrix = dist_mat
    
    def get_adjacency_matrix(self):
        return self.adjacency_matrix
    def get_covariance_matrix(self):
        return self.covariance_matrix
    def get_squeezing_value(self):
        return self.squeezing_value
    def get_num_nodes(self):
        return self.num_nodes
    def get_normalization_type(self):
        return self.normalization_type
    def get_sympeigvals_alice(self):
        return self.symp_eig_alice
    
    def show_covariance_matrix(self):
        plt.matshow(self.covariance_matrix)
        plt.show()
    
    def show_graph(self):
        nx.draw(self.graph, with_labels=True, node_color='skyblue', node_size=100)
        plt.title("Quantum Network Graph")
        plt.show()

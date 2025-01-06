import numpy as np
from scipy.linalg import expm
from itertools import product

# Adapted from https://github.com/CQuIC/pysme/blob/master/src/pysme/gellmann.py

class Basis:
    def __init__(self, cluster, method = "Gell-Mann"):
        """
        Initialize the quantum simulator with a graph and parameters.
        """
        self.method = method
        self.cluster = cluster

    def initialize_basis(self):
        """
        Initialize the quantum state and covariance matrix based on the squeezing value.
        """
        
        if self.method == "Gell-Mann":
            self.Gell_Mann_generation()
        
        else:
            raise ValueError("Unknown method")
        
    @staticmethod
    def gellmann(j, k, d):
        """
        Returns a generalized Gell-Mann matrix of dimension d. According to the
        convention in *Bloch Vectors for Qubits* by Bertlmann and Krammer (2008),
        returns :math:`\Lambda^j` for :math:`1\leq j=k\leq d-1`,
        :math:`\Lambda^{kj}_s` for :math:`1\leq k<j\leq d`,
        :math:`\Lambda^{jk}_a` for :math:`1\leq j<k\leq d`, and
        :math:`I` for :math:`j=k=d`.
        :param j: First index for generalized Gell-Mann matrix
        :type j:  positive integer
        :param k: Second index for generalized Gell-Mann matrix
        :type k:  positive integer
        :param d: Dimension of the generalized Gell-Mann matrix
        :type d:  positive integer
        :returns: A genereralized Gell-Mann matrix.
        :rtype:   numpy.array
        """

        if j > k:
            gjkd = np.zeros((d, d), dtype=np.complex128)
            gjkd[j - 1][k - 1] = 1
            gjkd[k - 1][j - 1] = 1
        elif k > j:
            gjkd = np.zeros((d, d), dtype=np.complex128)
            gjkd[j - 1][k - 1] = -1.j
            gjkd[k - 1][j - 1] = 1.j
        elif j == k and j < d:
            gjkd = np.sqrt(2/(j*(j + 1)))*np.diag([1 + 0.j if n <= j
                                               else (-j + 0.j if n == (j + 1)
                                                     else 0 + 0.j)
                                               for n in range(1, d + 1)])
        else:
            gjkd = np.diag([1 + 0.j for n in range(1, d + 1)])

        return gjkd
    
    def build_basis(self, d):
        '''
        Return a basis of orthogonal Hermitian operators on a Hilbert space of
        dimension d, with the identity element in the last place.
        '''
        return np.array([self.gellmann(j, k, d) for j, k in product(range(1, d + 1), repeat=2)])
    
    def Gell_Mann_generation(self):
        """
        Generate the basis for optimization.
        """
        self.basis_alice = self.build_basis(self.cluster.get_num_nodes()//2) #Basis for Alice, Number of nodes for Alice
        self.basis_bob = self.basis_alice #Basis for Bob, assuming equal number of modes for Alice and Bob
        
    def get_basis_alice(self):
        """
        Return the basis for Alice.
        """
        return self.basis_alice
    def get_basis_bob(self):
        """
        Return the basis for Bob.
        """
        return self.basis_bob
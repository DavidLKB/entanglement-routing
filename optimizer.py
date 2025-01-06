import numpy as np
from numpy.linalg import matrix_power, norm
from scipy.linalg import expm
import matplotlib.pyplot as plt
from numpy import linalg as la

class Optimizer:
    def __init__(self, cluster, basis):
        self.cluster = cluster
        self.basis = basis

    def initialize_optimizer(self, generation_number=10000, objective_function="standard", alice_node_number=0, bob_node_number=3):
        self.generation_number = generation_number
        self.alice_node_number = alice_node_number
        self.bob_node_number = bob_node_number

        self.num_nodes_provider_alice = self.cluster.get_num_nodes()//2
        self.num_nodes_provider_bob = self.num_nodes_provider_alice
        self.dim = (self.num_nodes_provider_alice)**2 + (self.num_nodes_provider_bob)**2

        self.l = int(4 + np.floor(3 * np.log(self.dim)))
        self.mu = int(np.floor(self.l / 2))
        self.sigma = 0.33

        weights = np.array([np.log(self.mu + 1) / k for k in range(1, self.mu + 1)])
        self.weights = weights / np.sum(weights)
        self.mueff = 1. / np.sum(self.weights ** 2)

        self.cc = 4. / (self.dim + 4)
        self.cs = (self.mueff + 2.) / (self.dim + self.mueff + 3.)
        self.ccov1 = min(1, (1.5 * self.dim) / 3 * (2 / (self.mueff + (1.3 + self.dim) ** 2)))
        self.ccovmu = min(1. - 2. / (self.mueff + (1.3 + self.dim) ** 2), ((1.5 + self.dim) / 3) * 
                          2 * (self.mueff - 2 + 1. / self.mueff) / (self.mueff + (self.dim + 2) ** 2))
        self.chiN = np.sqrt(self.dim) * (1 - 1. / (4 * self.dim) + 1. / (21 * self.dim ** 2))
        self.ds = 1 + self.cs + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1)

        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.Cov = np.ones(self.dim)
        self.Diag = np.ones(self.dim)

        self.xmean = np.random.rand(self.dim)
        self.bestFitness = np.zeros(self.generation_number)
        self.initialize_cov_ideal()
        self.init_objective_function()

    @staticmethod
    def Cov_ideal(lamb, mu, noeud_1, noeud_2, n):
        M = np.zeros((4, 2 * n))
        M[0, noeud_1], M[1, noeud_2], M[2, noeud_1 + n], M[3, noeud_2 + n] = [lamb] * 4
        M[0, noeud_2 + n], M[1, noeud_1 + n], M[2, noeud_2], M[3, noeud_1] = [mu] * 4
        return M

    def initialize_cov_ideal(self):
        n = self.cluster.get_num_nodes()
        self.cov_ideal = self.Cov_ideal(self.cluster.lambda_val, self.cluster.mu_val,
                                        self.alice_node_number, self.bob_node_number, n)

    def trace_out_ideal(self, cov):
        n1 = self.alice_node_number
        n2 = self.bob_node_number
        n = self.cluster.get_num_nodes()
        M = np.zeros((4,4))
        M[0,0],M[0,1],M[0,2],M[0,3] = cov[0,n1],cov[0,n2],cov[0,n1+n],cov[0,n2+n]
        M[1,0],M[1,1],M[1,2],M[1,3] = cov[1,n1],cov[1,n2],cov[1,n1+n],cov[1,n2+n]
        M[2,0],M[2,1],M[2,2],M[2,3] = cov[2,n1],cov[2,n2],cov[2,n1+n],cov[2,n2+n]
        M[3,0],M[3,1],M[3,2],M[3,3] = cov[3,n1],cov[3,n2],cov[3,n1+n],cov[3,n2+n]
        return M

    @staticmethod
    def generator_GM(par, basis, n):
        A = np.tensordot(par, basis, axes=1)
        return expm(1j * A)

    def Sym(self, par):
        zero_1 = np.zeros((self.num_nodes_provider_alice, self.num_nodes_provider_bob))
        zero_2 = np.zeros((self.num_nodes_provider_bob, self.num_nodes_provider_alice))
        U_A = self.generator_GM(par[:self.num_nodes_provider_alice ** 2], self.basis.basis_alice, self.num_nodes_provider_alice)
        X_A, Y_A = np.real(U_A), np.imag(U_A)
        U_B = self.generator_GM(par[self.num_nodes_provider_bob ** 2:], self.basis.basis_bob, self.num_nodes_provider_bob)
        X_B, Y_B = np.real(U_B), np.imag(U_B)
        return np.block([[X_A, zero_1, -Y_A, zero_1],
                         [zero_2, X_B, zero_2, -Y_B],
                         [Y_A, zero_1, X_A, zero_1],
                         [zero_2, Y_B, zero_2, X_B]])
        
    @staticmethod
    def compute_purity(A):
        return 1/np.sqrt(np.linalg.det(A))

    def init_objective_function(self):
        self.val_alpha = 0.5
        def fopt_test_pur(par):
            sym = self.Sym(par)
            n = self.cluster.get_num_nodes()
            mat = sym @ self.cluster.covariance_matrix @ sym.T
            cov_cluster2 = np.block([[mat[self.alice_node_number, :]], [mat[self.bob_node_number, :]], 
                                     [mat[self.alice_node_number + n, :]], [mat[self.bob_node_number + n, :]]])
            a = self.trace_out_ideal(cov_cluster2)
            return (norm(self.cov_ideal - cov_cluster2, ord='fro') + self.val_alpha * (1 - self.compute_purity(a)))
        self.funct = fopt_test_pur

    def optimize(self):
        for i in range(1, self.generation_number + 1):
            if i%100==0:
                print('starting generation ', i, ' of total', self.generation_number)
            tries = 0
            while tries < 10:
                offspring = np.zeros((self.l, self.dim))
                mutations = np.zeros((self.l, self.dim))
                for j in range(self.l):
                    znew = np.random.randn(self.dim)
                    offspring[j, :] = self.xmean + self.sigma * self.Diag * znew
                    mutations[j, :] = znew
                try:
                    fitness = np.apply_along_axis(self.funct, 1, offspring)
                except np.linalg.LinAlgError:
                    tries += 1
                else:
                    break
            best = np.min(fitness)
            self.bestFitness[i - 1] = best
            idx = np.argsort(fitness)
            bestIndividuals = offspring[idx][:self.mu]
            bestMutants = mutations[idx][:self.mu]
            
            if i%100==0:
                print("Objective function: " + str(best))
                sym = self.Sym(bestIndividuals[0])
                n = self.cluster.get_num_nodes()
                mat = sym @ self.cluster.covariance_matrix @ sym.T
                cov_cluster2 = np.block([[mat[self.alice_node_number, :]], [mat[self.bob_node_number, :]], 
                                     [mat[self.alice_node_number + n, :]], [mat[self.bob_node_number + n, :]]])
                print("Purity: " + str(self.compute_purity(self.trace_out_ideal(cov_cluster2))))
            
            self.xmean = np.dot(bestIndividuals.T, self.weights)
            self.zmean = np.dot(bestMutants.T , self.weights)
            self.ps = (1 - self.cs) * self.ps + np.sqrt( self.cs * (2 - self.cs) * self.mueff) * self.zmean
            
            # update of the global step-size
            self.sigma = self.sigma * np.exp( ( self.cs / self.ds ) * ( la.norm( self.ps )/self.chiN - 1))
            self.hsig = (la.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2*i)) ) < self.chiN * (1.4 + 2./(self.dim + 1.))
        
            # update the evolution path
            self.pc = (1 - self.cc) * self.pc + self.hsig * np.sqrt( self.cc * (2 - self.cc) * self.mueff ) * self.Diag * self.zmean      
            
            # update of covariance matrix Cov
            self.Cov = (1 - self.ccov1 - self.ccovmu) * self.Cov + ( self.ccov1 *  self.pc**2 ) # a checker  pour le carré sur pc       
            
            # eigendecompose the new covariance matrix
            self.Diag = np.sqrt(self.Cov)
            
            # optimization routine ends
       
        # returns the the last set of parameters and the corresponding fitness.    
        self.solution, self.solution_fitness = bestIndividuals[0],self.bestFitness[-1]

            
            
    def get_solution(self):
        return self.xmean

    def show_solution(self):
        n = self.cluster.get_num_nodes()
        sym = self.Sym(self.xmean)
        mat = sym @ self.cluster.covariance_matrix @ sym.T
        cov_cluster2 = np.block([[mat[self.alice_node_number,:]],[mat[self.bob_node_number,:]],
                                 [mat[self.alice_node_number+n,:]],[mat[self.bob_node_number+n,:]]])
        # Créer la figure et les axes
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

        # Tracer la première figure
        ax1.matshow(cov_cluster2)
        ax1.set_title('optimized covariance matrix')

        # Tracer la deuxième figure
        ax2.matshow(self.cov_ideal)
        ax2.set_title('ideal covariance matrix')
        
        ax3.matshow(self.cov_ideal - cov_cluster2)
        ax3.set_title('difference between the two matrices')
        # Ajuster l'espacement entre les figures
        # plt.tight_layout()

        # Afficher les figures
        plt.show()
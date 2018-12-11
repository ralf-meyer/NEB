import numpy as np
import tensorflow as tf
from DescriptorLib.SymmetryFunctionSet import SymmetryFunctionSet
from NNpotentials import BPpotential
from NNpotentials.utils import calculate_bp_indices
from itertools import combinations_with_replacement

class NNModel(object):

    def __init__(self, molecule, C1 = 1.0, C2 = 1.0, layers = None,
        reset_fit = True, normalize_input = True):
        self.molecule = molecule
        self.C1 = C1
        self.C2 = C2
        self.reset_fit = reset_fit
        self.normalize_input = normalize_input
        self.session = tf.Session()

        self.Gs_mean = {}
        self.Gs_std = {}

        if self.molecule == 'Ethane':

            self.types = ['H', 'H', 'H', 'C', 'C', 'H', 'H', 'H']
            self.unique_types = ['C', 'H']
            offset_dict = {'C': -37.0895866384, 'H': -0.4665818496}
        elif self.molecule == 'Ammonia':
            self.types = ['N', 'H', 'H', 'H']
            self.unique_types = ['N', 'H']
            offset_dict = {'N': -54.2562426504, 'H': -0.4665818496}

        self.sfs = SymmetryFunctionSet(self.unique_types, cutoff = 6.5)
        self.int_types = [self.sfs.type_dict[ti] for ti in self.types]

        # Parameters from Artrith and Kolpak Nano Lett. 2014, 14, 2670
        etas = [0.0009, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2]
        for t1 in self.sfs.atomtypes:
            for t2 in self.sfs.atomtypes:
                for eta in etas:
                    self.sfs.add_TwoBodySymmetryFunction(t1, t2, 'BehlerG1',
                        [eta], cuttype='cos')

        ang_etas = [0.0001, 0.003, 0.008]
        zetas = [1.0, 4.0]
        for ti in self.sfs.atomtypes:
            for (tj, tk) in combinations_with_replacement(
                self.sfs.atomtypes, 2):
                for etas in ang_etas:
                    for lamb in [-1.0, 1.0]:
                        for zeta in zetas:
                            self.sfs.add_ThreeBodySymmetryFunction(
                                ti, tj, tk, "BehlerG3", [lamb, zeta, eta],
                                cuttype = 'cos')

        self.pot = BPpotential(self.unique_types,
            [self.sfs.num_Gs[self.sfs.type_dict[t]] for t in self.unique_types],
            layers = [[5, 5]]*len(self.unique_types), build_forces = True,
            offsets = [offset_dict[t] for t in self.unique_types],
            precision = tf.float64)

        for t in self.sfs.atomtypes:
            self.Gs_mean[t] = np.zeros(self.sfs.num_Gs[self.sfs.type_dict[t]])
            self.Gs_std[t] = np.ones(self.sfs.num_Gs[self.sfs.type_dict[t]])

        self.session.run(tf.initializers.variables(self.pot.variables))

    def fit(self, x_train, y_train, x_prime_train, y_prime_train):
        """
        Fitting a new model to a given training pattern
        :param x_train: function input pattern shape = [N_samples, N_features]
        :param y_train: function values shape = [N_samples, 1]
        :param x_prime_train: derivative input pattern shape = [N_samples, N_features]
        :param y_prime_train: derivative values shape = [N_samples, N_features]
        :return:
        """

        print('fit called with %d geometries. E_max = %f, E_min = %f'%(
            len(x_train), np.max(y_train), np.min(y_train)))
        # NN Model does not support different geometries for energies and forces
        np.testing.assert_array_equal(x_train, x_prime_train)

        xyzs = x_train.reshape((-1, len(self.types), 3))

        Gs = []
        dGs = []

        for i in range(len(xyzs)):
            Gi, dGi = self.sfs.eval_with_derivatives(self.types, xyzs[i,:,:])
            Gs.append(Gi)
            dGs.append(dGi)
        ANN_inputs, indices, ANN_derivs = calculate_bp_indices(
            len(self.unique_types), Gs, [self.int_types]*len(Gs), dGs = dGs)
        if self.normalize_input:
            for i, t in enumerate(self.unique_types):
                self.Gs_mean[t] = np.mean(ANN_inputs[i], axis = 0)
                # Small offset for numerical stability
                self.Gs_std[t] = np.std(ANN_inputs[i], axis = 0) + 1E-6
        train_dict = {self.pot.target: y_train,
            self.pot.target_forces: -y_prime_train.reshape(
                (-1, len(self.types), 3)),
            self.pot.rmse_weights: np.ones(len(Gs))}
        for i, t in enumerate(self.unique_types):
            train_dict[self.pot.ANNs[t].input] = (
                ANN_inputs[i]-self.Gs_mean[t])/self.Gs_std[t]
            train_dict[self.pot.atom_indices[t]] = indices[i]
            train_dict[self.pot.ANNs[t].derivatives_input] = np.einsum(
                'ijkl,j->ijkl', ANN_derivs[i], 1.0/self.Gs_std[t])

        if self.reset_fit:
            self.session.run(tf.initializers.variables(self.pot.variables))
        regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(
            regularizer, reg_variables)
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.C1*self.pot.rmse + self.C2*self.pot.rmse_forces + reg_term,
            method='L-BFGS-B')
        optimizer.minimize(self.session, train_dict)
        e_rmse, f_rmse = self.session.run(
            [self.pot.rmse, self.pot.rmse_forces], train_dict)
        print('fit finished with energy rmse '
            '%f and gradient rmse %f'%(e_rmse, f_rmse))

    def predict(self, x):
        xyzs = x.reshape((-1,len(self.types),3))
        Gs = []
        for i in range(len(xyzs)):
            Gs.append(self.sfs.eval(self.types, xyzs[i,:,:]))
        ANN_inputs, indices, = calculate_bp_indices(
            len(self.unique_types), Gs, [self.int_types]*len(Gs))
        eval_dict = {self.pot.target: np.zeros(len(Gs))}
        for i, t in enumerate(self.unique_types):
            eval_dict[self.pot.ANNs[t].input] = (
                ANN_inputs[i]-self.Gs_mean[t])/self.Gs_std[t]
            eval_dict[self.pot.atom_indices[t]] = indices[i]

        return self.session.run(self.pot.E_predict, eval_dict)

    def predict_derivative(self, x):
        xyzs = x.reshape((-1, len(self.types), 3))
        Gs = []
        dGs = []
        for i in range(len(xyzs)):
            Gi, dGi = self.sfs.eval_with_derivatives(self.types, xyzs[i,:,:])
            Gs.append(Gi)
            dGs.append(dGi)
        ANN_inputs, indices, ANN_derivs = calculate_bp_indices(
            len(self.unique_types), Gs, [self.int_types]*len(Gs), dGs = dGs)
        eval_dict = {self.pot.target: np.zeros(len(Gs)),
            self.pot.target_forces: np.zeros((len(Gs), len(self.types), 3))}
        for i, t in enumerate(self.unique_types):
            eval_dict[self.pot.ANNs[t].input] = (
                ANN_inputs[i]-self.Gs_mean[t])/self.Gs_std[t]
            eval_dict[self.pot.atom_indices[t]] = indices[i]
            eval_dict[self.pot.ANNs[t].derivatives_input] = np.einsum(
                'ijkl,j->ijkl', ANN_derivs[i], 1.0/self.Gs_std[t])

        # Return gradient (negative force)
        return -self.session.run(self.pot.F_predict, eval_dict).reshape(-1)


    def predict_val_der(self, x, *args):
        """
        function to allow to use the implemented NEB method
        :param x: prediction pattern, for derivative and function value the same shape = [N_samples, N_features]
        :return: function value prediction, derivative prediction
        """
        xyzs = x.reshape((-1, len(self.types), 3))
        Gs = []
        dGs = []
        for i in range(len(xyzs)):
            Gi, dGi = self.sfs.eval_with_derivatives(self.types, xyzs[i,:,:])
            Gs.append(Gi)
            dGs.append(dGi)
        ANN_inputs, indices, ANN_derivs = calculate_bp_indices(
            len(self.unique_types), Gs, [self.int_types]*len(Gs), dGs = dGs)
        eval_dict = {self.pot.target: np.zeros(len(Gs)),
            self.pot.target_forces: np.zeros((len(Gs), len(self.types), 3))}
        for i, t in enumerate(self.unique_types):
            eval_dict[self.pot.ANNs[t].input] = (
                ANN_inputs[i]-self.Gs_mean[t])/self.Gs_std[t]
            eval_dict[self.pot.atom_indices[t]] = indices[i]
            eval_dict[self.pot.ANNs[t].derivatives_input] = np.einsum(
                'ijkl,j->ijkl', ANN_derivs[i], 1.0/self.Gs_std[t])

        E = self.session.run(self.pot.E_predict, eval_dict)
        F = self.session.run(self.pot.F_predict, eval_dict)
        # Return energey and gradient (negative force)
        return E, -F.reshape(-1), None

    def close(self):
        self.sfs.close()
        self.session.close()

from additional import *
import numpy as np
import pandas as pd


class Model:
    samples_filename = None
    output_filename = None

    X_df, X_array, X_normed_df, X_normed_array = None, None, None, None
    Y_df, Y_array, Y_normed_df, Y_normed_array = None, None, None, None

    Q, X_dim, Y_dim = 0, 0, 0

    X_dims, X1_dim, X2_dim, X3_dim = None, 0, 0, 0
    P_dims, P1_dim, P2_dim, P3_dim = None, 0, 0, 0

    b_q = None
    F, lambda01, lambda02, lambda03 = None, None, None, None
    psi, a0, c0 = None, None, None
    polynomials, results_normed, results = None, None, None
    residuals = None

    RESIDUALS_LOG = ''
    LOG = ''

    polynomials_dictionary = {
        Polynom.CHEBYSHEV.name: [chebyshev_bias_value, chebyshev_bias_polynomial],
        Polynom.LEGANDRE.name: [legandre_value, legandre_polynomial],
        Polynom.LAGERR.name: [lagerr_value, lagerr_polynomial]
    }

    solve_method = conjugate_gradient_method
    polynomial_value_method = polynomials_dictionary[Polynom.CHEBYSHEV.name][0]
    polynomial_function_method = polynomials_dictionary[Polynom.CHEBYSHEV.name][1]
    weight_method = Weight.NORMED
    lambda_method = Lambda.SINGLE_SET

    def __init__(self, s_f, o_f, P_dims, polynomial, weight, lambdas):
        self.samples_filename = s_f
        self.output_filename = o_f

        samples = pd.read_excel(self.samples_filename, index_col=0)
        self.Q = len(samples.index)
        for column in samples.columns:
            variable_name = column[0]
            index = column[1]
            if variable_name == 'x':
                if index == '1':
                    self.X1_dim += 1
                elif index == '2':
                    self.X2_dim += 1
                elif index == '3':
                    self.X3_dim += 1
            elif variable_name == 'y':
                self.Y_dim += 1
        self.X_dims = np.array([self.X1_dim, self.X2_dim, self.X3_dim])
        self.X_dim = len(self.X_dims)

        self.X_df = samples.iloc[:, :sum(self.X_dims)]
        self.Y_df = samples.iloc[:, sum(self.X_dims):]

        self.X_array = self.X_df.to_numpy()
        self.Y_array = self.Y_df.to_numpy()

        self.X_normed_df = (self.X_df - self.X_df.min()) / (self.X_df.max() - self.X_df.min())
        self.Y_normed_df = (self.Y_df - self.Y_df.min()) / (self.Y_df.max() - self.Y_df.min())

        self.X_normed_array = self.X_normed_df.to_numpy()
        self.Y_normed_array = self.Y_normed_df.to_numpy()

        self.P_dims = P_dims
        self.P1_dim, self.P2_dim, self.P3_dim = [self.P_dims[i] for i in range(len(P_dims))]

        self.polynomial_value_method = self.polynomials_dictionary[polynomial][0]
        self.polynomial_function_method = self.polynomials_dictionary[polynomial][1]
        self.weight_method = weight
        self.lambda_method = lambdas

        self.b_q = ((self.Y_df.max(axis=1) + self.Y_df.min(axis=1)) / 2).to_numpy()

    def get_F(self):
        self.F = np.zeros((self.Q, np.sum(self.X_dims * (self.P_dims + 1))))
        for q in range(self.Q):
            width_index = 0
            for i in range(len(self.X_dims)):
                for j in range(self.X_dims[i]):
                    for p in range(self.P_dims[i] + 1):
                        self.F[q, width_index] = self.polynomial_value_method(
                            self.X_normed_df.iloc[q][f'x{i + 1}{j + 1}'], p)
                        width_index += 1

    def get_lambdas_generalized(self, b_q):
        b_q = np.reshape(b_q, (-1, 1))

        self.lambda01 = np.zeros((self.X1_dim, self.P1_dim + 1))
        self.lambda02 = np.zeros((self.X2_dim, self.P2_dim + 1))
        self.lambda03 = np.zeros((self.X3_dim, self.P3_dim + 1))
        if self.lambda_method == Lambda.TRIPLE_SET.name:
            F1 = self.F[:, :self.X1_dim * (self.P1_dim + 1)]
            F2 = self.F[:,
                 self.X1_dim * (self.P1_dim + 1): self.X1_dim * (self.P1_dim + 1) + self.X2_dim * (self.P2_dim + 1)]
            F3 = self.F[:, self.X1_dim * (self.P1_dim + 1) + self.X2_dim * (self.P2_dim + 1): self.X1_dim * (
                    self.P1_dim + 1) + self.X2_dim * (self.P2_dim + 1) + self.X3_dim * (self.P3_dim + 1)]

            self.lambda01 = np.reshape(self.solve_method(F1, b_q), (self.X1_dim, self.P1_dim + 1))
            self.lambda02 = np.reshape(self.solve_method(F2, b_q), (self.X2_dim, self.P2_dim + 1))
            self.lambda03 = np.reshape(self.solve_method(F3, b_q), (self.X3_dim, self.P3_dim + 1))
        else:
            lambda0 = self.solve_method(self.F, b_q)
            for i in range(self.X1_dim):
                self.lambda01[i] = lambda0[i * (self.P1_dim + 1): (i + 1) * (self.P2_dim + 1)].T
            for i in range(self.X2_dim):
                self.lambda02[i] = lambda0[i * (self.P2_dim + 1) + self.X1_dim * (self.P1_dim + 1): (i + 1) * (
                        self.P2_dim + 1) + self.X1_dim * (self.P1_dim + 1)].T
            for i in range(self.X3_dim):
                self.lambda03[i] = lambda0[i * (self.P3_dim + 1) + self.X2_dim * (self.P2_dim + 1) + self.X2_dim * (
                        self.P2_dim + 1): (i + 1) * (self.P3_dim + 1) + self.X1_dim * (
                        self.P1_dim + 1) + self.X2_dim * (self.P2_dim + 1)].T

    def get_psi(self):
        self.psi = np.zeros((self.Q, np.sum(self.X_dims)))
        for q in range(self.Q):
            for j in range(np.sum(self.X_dims)):
                if j < self.X1_dim:
                    self.psi[q, j] = self.lambda01[j] @ self.F[q, j * (self.P1_dim + 1): (j + 1) * (self.P2_dim + 1)].T
                elif j < self.X1_dim + self.X2_dim:
                    j_cor = j - self.X1_dim
                    shift = (self.P1_dim + 1) * self.X1_dim
                    self.psi[q, j] = self.lambda02[j_cor] @ self.F[q, j_cor * (self.P2_dim + 1) + shift: (j_cor + 1) * (
                            self.P2_dim + 1) + shift].T
                else:
                    j_cor = j - self.X1_dim - self.X2_dim
                    shift = (self.P1_dim + 1) * self.X1_dim + (self.P2_dim + 1) * self.X2_dim
                    self.psi[q, j] = self.lambda03[j_cor] @ self.F[q, j_cor * (self.P3_dim + 1) + shift: (j_cor + 1) * (
                            self.P3_dim + 1) + shift].T

    def get_a(self):
        self.a0 = np.zeros((self.Y_dim, np.sum(self.X_dims)))
        for f in range(self.Y_dim):
            a01 = self.solve_method(self.psi[:, :self.X1_dim], self.Y_normed_array[:, f].reshape(-1, 1))

            a02 = self.solve_method(self.psi[:, self.X1_dim: self.X1_dim + self.X2_dim],
                                    self.Y_normed_array[:, f].reshape(-1, 1))

            a03 = self.solve_method(self.psi[:, self.X1_dim + self.X2_dim:], self.Y_normed_array[:, f].reshape(-1, 1))

            self.a0[f, :] = np.concatenate((a01, a02, a03)).T

    def get_c(self):
        self.c0 = np.zeros((self.Y_dim, self.X_dim))
        for i in range(self.Y_dim):
            Fij = np.zeros((self.Q, self.X_dim))
            for q in range(self.Q):
                Fij[q, 0] = self.a0[i, :self.X1_dim].reshape(1, -1) @ self.psi[q, :self.X1_dim].reshape(-1, 1)
                Fij[q, 1] = self.a0[i, self.X1_dim: self.X1_dim + self.X2_dim].reshape(1, -1) @ self.psi[q,
                                                                                                self.X1_dim: self.X1_dim + self.X2_dim].reshape(
                    -1, 1)
                Fij[q, 2] = self.a0[i, self.X1_dim + self.X2_dim:].reshape(1, -1) @ self.psi[q,
                                                                                    self.X1_dim + self.X2_dim:].reshape(
                    -1,
                    1)
            self.c0[i] = self.solve_method(Fij, self.Y_normed_array[:, i].reshape(-1, 1)).T

    def get_restored_functions(self):
        c_va_lambda01 = np.zeros((self.Y_dim, self.X1_dim * (self.P1_dim + 1)))
        c_va_lambda02 = np.zeros((self.Y_dim, self.X2_dim * (self.P2_dim + 1)))
        c_va_lambda03 = np.zeros((self.Y_dim, self.X3_dim * (self.P3_dim + 1)))
        polynomials = np.zeros((self.Y_dim, np.sum(self.X_dims * (self.P_dims + 1))))

        for f in range(self.Y_dim):
            a_lambda01 = np.zeros((self.X1_dim, self.P1_dim + 1))
            for i in range(self.X1_dim):
                a_lambda01[i] = self.a0[f, i] * self.lambda01[i]
            va_lambda01 = a_lambda01.flatten()

            a_lambda02 = np.zeros((self.X2_dim, self.P2_dim + 1))
            for i in range(self.X2_dim):
                a_lambda02[i] = self.a0[f, i + self.X1_dim] * self.lambda02[i]
            va_lambda02 = a_lambda02.flatten()

            a_lambda03 = np.zeros((self.X3_dim, self.P3_dim + 1))
            for i in range(self.X3_dim):
                a_lambda03[i] = self.a0[f, i + self.X1_dim + self.X2_dim] * self.lambda03[i]
            va_lambda03 = a_lambda03.flatten()

            c_va_lambda01[f] = self.c0[f, 0] * va_lambda01
            c_va_lambda02[f] = self.c0[f, 1] * va_lambda02
            c_va_lambda03[f] = self.c0[f, 2] * va_lambda03

            polynomials[f] = np.concatenate((c_va_lambda01[f], c_va_lambda02[f], c_va_lambda03[f]))
        return polynomials

    def print_restored_functions(self, index=-1):
        log = 'Відновлені через поліноми функції:\n\n'
        if index == -1:
            for f in range(self.Y_dim):
                log += f"Ф{f + 1} (x1, x2, x3) ="
                for j in range(self.X1_dim):
                    if j > 0:
                        log += '\t\t'
                    for k in range(self.P1_dim + 1):
                        starter = " " if j == 0 and k == 0 else " +"
                        log += f"{starter} {self.polynomials[f, (self.P1_dim + 1) * j + k]:.4f} T{k}(x1{j + 1})"
                    log += '\n'
                shift = self.X1_dim * (self.P1_dim + 1)
                for j in range(self.X2_dim):
                    log += '\t\t'
                    for k in range(self.P2_dim + 1):
                        log += f" + {self.polynomials[f, (self.P2_dim + 1) * j + k + shift]:.4f} T{k}(x2{j + 1})"
                    log += '\n'
                shift += self.X2_dim * (self.P2_dim + 1)
                for j in range(self.X3_dim):
                    log += '\t\t'
                    for k in range(self.P3_dim + 1):
                        log += f" + {self.polynomials[f, (self.P3_dim + 1) * j + k + shift]:.4f} T{k}(x3{j + 1})"
                    log += '\n'
                log += '\n'
        else:
            log += f"Ф{index + 1} (x1, x2, x3) ="
            for j in range(self.X1_dim):
                if j > 0:
                    log += '\t\t'
                for k in range(self.P1_dim + 1):
                    starter = " " if j == 0 and k == 0 else " +"
                    log += f"{starter} {self.polynomials[index, (self.P1_dim + 1) * j + k]:.4f} T{k}(x1{j + 1})"
                log += '\n'
            shift = self.X1_dim * (self.P1_dim + 1)
            for j in range(self.X2_dim):
                log += '\t\t'
                for k in range(self.P2_dim + 1):
                    log += f" + {self.polynomials[index, (self.P2_dim + 1) * j + k + shift]:.4f} T{k}(x2{j + 1})"
                log += '\n'
            shift += self.X2_dim * (self.P2_dim + 1)
            for j in range(self.X3_dim):
                log += '\t\t'
                for k in range(self.P3_dim + 1):
                    log += f" + {self.polynomials[index, (self.P3_dim + 1) * j + k + shift]:.4f} T{k}(x3{j + 1})"
                log += '\n'
            log += '\n'
        return log

    def get_normed_functions(self):
        results_normed = np.zeros((self.Q, self.Y_dim))
        for q in range(self.Q):
            for f in range(self.Y_dim):
                for j in range(self.X1_dim):
                    for k in range(self.P1_dim + 1):
                        results_normed[q, f] += self.polynomials[f, (self.P1_dim + 1) * j + k] * self.F[
                            q, (self.P1_dim + 1) * j + k]
                shift = self.X1_dim * (self.P1_dim + 1)
                for j in range(self.X2_dim):
                    for k in range(self.P2_dim + 1):
                        results_normed[q, f] += self.polynomials[f, (self.P2_dim + 1) * j + k + shift] * self.F[
                            q, (self.P2_dim + 1) * j + k + shift]
                shift += self.X2_dim * (self.P2_dim + 1)
                for j in range(self.X3_dim):
                    for k in range(self.P3_dim + 1):
                        results_normed[q, f] += self.polynomials[f, (self.P3_dim + 1) * j + k + shift] * self.F[
                            q, (self.P3_dim + 1) * j + k + shift]
        return results_normed

    def get_functions(self):
        mins = self.Y_array.min(axis=0)
        maxs = self.Y_array.max(axis=0)
        self.results = np.array(
            [[self.results_normed[q][i] * (maxs[i] - mins[i]) + mins[i] for i in range(self.Y_dim)] for q in
             range(self.Q)])

    def calculate(self):
        self.get_F()
        self.polynomials = np.zeros((self.Y_dim, np.sum(self.X_dims * (self.P_dims + 1))))
        self.results_normed = np.zeros((self.Q, self.Y_dim))

        if self.weight_method == Weight.NORMED.name:
            self.RESIDUALS_LOG += 'Нев\'язка:\n'
            for i in range(self.Y_dim):
                self.get_lambdas_generalized(self.Y_normed_array[:, i]) # OK
                self.get_psi() # Most likely OK
                self.get_a() # Most likely OK
                self.get_c() # Most likely OK
                self.polynomials[i] = self.get_restored_functions()[i]
                self.results_normed[:, i] = self.get_normed_functions()[:, i]

                self.LOG += f'Для Ф{i + 1} (x1, x2, x3):\n\nЛямбда 1:\n{self.lambda01}\n\nЛямбда 2:\n{self.lambda02}\n\nЛямбда 3:\n{self.lambda03}\n\n' + \
                            f'Псі:\n{self.psi}\n\nА:\n{self.a0}\n\nC:\n{self.c0}\n\n'
                self.RESIDUALS_LOG += f'Ф{i + 1}: {np.max(np.abs(self.Y_normed_array[:, i] - self.results_normed[:, i])):.6f}\n'
        else:
            self.get_lambdas_generalized(self.b_q)
            self.get_psi()
            self.get_a()
            self.get_c()
            self.polynomials = self.get_restored_functions()
            self.results_normed = self.get_normed_functions()

            LOG = f'Лямбда 1:\n{self.lambda01}\n\nЛямбда 2:\n{self.lambda02}\n\nЛямбда 3:\n{self.lambda03}\n\n' + \
                  f'Псі:\n{self.psi}\n\nА:\n{self.a0}\n\nC:\n{self.c0}\n\n'
            self.RESIDUALS_LOG += 'Нев\'язка:\n'
            for j in range(self.Y_dim):
                self.RESIDUALS_LOG += f"Ф{j + 1}: {np.max(np.abs(self.Y_normed_array[:, j] - self.results_normed[:, j])):.6f}\n"

        self.get_functions()
        self.residuals = [f'{np.max(np.abs(self.Y_normed_array[:, i] - self.results_normed[:, i])):.6f}' for i in
                          range(len(self.results_normed[0]))]

        self.LOG += self.print_restored_functions()
        self.LOG += self.RESIDUALS_LOG

        results_file = open(self.output_filename + '.txt', 'w', encoding="utf-8")
        results_file.write(self.LOG)
        results_file.close()

import numpy as np

class GaussianProcess:
    def __init__(self, X_train, y_train, ktype, kernel_param, noise):
        """
        Initialize and fit the Gaussian Process.
        
        Parameters:
            X_train: array-like, shape (n_features, n_train)
            y_train: array-like, shape (, n_train)
            ktype: str, kernel type: 'rbf', 'polynomial', or 'linear'
            kernel_param: kernel parameter (e.g., inverse kernel width for rbf, degree for polynomial)
        """
        
        # # Check if the X_train is (n,d) shape, if not transpose it
        # if X_train.shape[0] < X_train.shape[1]:   
        #     X_train = X_train.T

        self.X_train = X_train
        self.y_train = y_train
        self.noise = noise

        # Set the kernel function based on ktype.
        if ktype == 'rbf':
            from rbfkernel import rbf_kernel
            self.kernel = lambda X, Y: rbf_kernel(X, Y, kpar=kernel_param)
        elif ktype == 'polynomial':
            from polynomialkernel import polynomial_kernel
            self.kernel = lambda X, Y: polynomial_kernel(X, Y, kpar=kernel_param)
        elif ktype == 'linear':
            from linearkernel import linear_kernel
            self.kernel = lambda X, Y: linear_kernel(X, Y)
        else:
            raise ValueError("Unsupported kernel type. Choose 'rbf', 'polynomial', or 'linear'.")
        
        self.fit()

    def fit(self):
        """
        Fit the GP to the training data by computing the kernel matrix and its Cholesky decomposition using matrix multiplications.
        This should also account for the noise parameter.
        """

        # self.K = None
        # self.alpha = None

        # Compute kernel matrix K(X_train, X_train)
        K = self.kernel(self.X_train, self.X_train)  # shape (n_train, n_train)
        # print(f"Kernel matrix shape: {K.shape}")

        # Add noise variance to the diagonal
        K += self.noise * np.eye(K.shape[0])

        # Cholesky decomposition: K + σ^2 I = L L^T
        self.L = np.linalg.cholesky(K)  # Lower triangular matrix

        # Solve for alpha using Cholesky: α = (K + σ^2 I)^-1 y
        # Step 1: L * v = y
        v = np.linalg.solve(self.L, self.y_train.T)

        # Step 2: L^T * alpha = v
        self.alpha = np.linalg.solve(self.L.T, v)

        

    def predict(self, X_test):
        """
        Make predictions at test points.
        
        Parameters:
            X_test: array-like, shape (d,n)
        
        Returns:
            mean: array, shape (n,), predictive mean
            variance: array, shape (n,), predictive variance
        """
        # YOUR CODE HERE

        # # Check if the X_test is (d,n) shape, if not transpose it
        # if X_test.shape[0] > X_test.shape[1]:
        #     X_test = X_test.T

        # Compute cross-kernel between test and training: K(X_test, X_train)
        K_s = self.kernel(X_test, self.X_train)  # shape (n_test, n_train)

        # Predictive mean: μ* = K_s @ alpha
        mean = K_s @ self.alpha  # shape (n_test,)

        # Solve L * v = K_s.T
        v = np.linalg.solve(self.L, K_s.T)  # shape (n_train, n_test)

        # Predictive variance: diag(K_ss - v^T v)
        K_ss = self.kernel(X_test, X_test)  # shape (n_test, n_test)
        variance = K_ss - v.T @ v  # shape (n_test, n_test)

        return mean, np.diag(variance)
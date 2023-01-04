import numpy as np

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10 * X**2) + 0.1 * np.sin(100 * X)


class LinearRegression:
    def __init__(self, basis='polynomial', J=1):
        self.J = J
        self.basis = basis

        self.mle_w = None
        self.mle_variance = None

    def design_matrix(self, X):
        if self.basis == 'polynomial':
            return self._polynomial_design_matrix(X)
        else:
            return self._trigonometric_design_matrix(X)
        
    def _polynomial_design_matrix(self, X):
        """ Return polynomial design matrix of degree J with shape (N, M)

            Args:
                X: input vector of shape (N, 1)

            Output: polynomial design matrix of shape (N, M)
        """

        # --- ENTER SOLUTION HERE ---
        col_size = self.J + 1
        row_size = X.shape[0]
        Phi = np.zeros((row_size, col_size)) # Initializing the return (polynomial design matrix)

        for row in range(row_size):
            for col in range(col_size):
                Phi[row, col] = X[row] ** col # Updating the value

        #print('Phi1 \n', Phi)

        return Phi

    def _trigonometric_design_matrix(self, X):
        """ Return trigonometric design matrix of degree J with shape (N, M)

            Args:
                X: input vector of shape (N, 1)

            Output: polynomial design matrix of shape (N, M)
        """

        # --- ENTER SOLUTION HERE ---
        col_size = 2 * self.J + 1
        row_size = X.shape[0]
        Phi = np.zeros((row_size, col_size))  # Initializing the return (trigonometric design matrix)

        for row in range(row_size):
            for col in range(col_size):

                if (col % 2) == 0: # Checking if we're on an even column (degree level : 0/2/4 = cos, 1/3/5 = sin)
                    Phi[row, col] = np.cos(2 * np.pi * (col//2) * X[row])  # Updating the value

                else:
                    Phi[row, col] = np.sin(2 * np.pi * ((col//2) + 1) * X[row])  # Updating the value

        #print('Phi2 \n', Phi)

        return Phi

    def fit(self, X, Y):
        """ Find maximum likelihood (MLE) solution, given basis Phi and output Y.

        Args:
            Phi: design matrix of shape (M, N)
            Y: vector of shape (N, 1)
            variance: scalar variance

        The function should not return anything, but instead
            1. save maximum likelihood for weights w, a numpy vector of shape (M, N), as variable 'self.mle_w'
            2. save maximum likelihood for variance as float as variable 'self.mle_variance'
        """

        # --- ENTER SOLUTION HERE ---
        Phi = self.design_matrix(X)
        w_param = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ Y  # Optimal parameters formula
        variance = np.std(Y - (Phi @ w_param)) ** 2  # Trying to have y = y_pred = (Theta.T @ Phi(X)) + eps and eps follows N(0, sigma²)

        #print('w_param \n', w_param)
        #print('mle_variance \n', variance)

        self.mle_w = w_param
        self.mle_variance = variance

    def predict(self, X_predict):
        """ Make a prediction using fitted solution.

        Args:
            X_predict: point to make prediction, vector of shape (V, 1)

        Output prediction as numpy vector of shape (V, 1)
        """
        
        # --- ENTER SOLUTION HERE ---
        # hint: remember that you can use functions like 'self.design_matrix(...)'
        #       and the fitted vector 'self.mle_w' here.

        Y_predict = self.design_matrix(X_predict) @ self.mle_w # Y_pred = vector of size (V, 1)

        #print('Y_predict \n', Y_predict)

        return Y_predict

    def predict_range(self, N_points, xmin, xmax):
        """ Make a prediction along a predefined range.

        Args:
            N_points: number of points to evaluate within range
            xmin: start of range to predict
            xmax: end of range to predict

        Returns a tuple containing:
            - numpy vector of shape (N_points, 1) for predicted X locations
            - numpy vector of shape (N_points, 1) for corresponding predicted values Y
        """

        # --- ENTER SOLUTION HERE ---
        X = np.linspace(xmin, xmax, N_points) # Creating an ndarray
        X_predict = np.reshape(X, (N_points, 1)) # Reshaping the ndarray to have a correct prediction shape
        Y_predict = self.predict(X_predict) # Perform prediction with matrix product defined above

        return X_predict, Y_predict


def leave_one_out_cross_validation(model, X, Y):
    """
    Function to perform leave-one-out cross validation.
    
    Args:
        model: Model to perform leave-one-out cross validation.
        X: Full dataset X, of which different folds should be made.
        Y: Labels of dataset X

    Should return two floats:
        - the average test error over different folds
        - the average mle variance over different folds
    """

    # --- ENTER SOLUTION HERE ---
    # Hint: use the functions 'model.fit()' to fit on train folds and
    #       the function 'model.predict() to predict on test folds.

    row_size = X.shape[0] # Number of elements (samples)
    test_errors = [] # Initializing the list that will contain the prediction error of each CV
    mle_variances = [] # Initializing the list that will contain the variance of each CV

    for sample in range(row_size):

        # Copying X and Y at each loop to have the full matrices
        #new_X = np.copy(X)
        #new_Y = np.copy(Y)

        # Deleting the element (leave one out)
        # X_sample = new_X[:,0].pop(sample) POP doesn't work on np.ndarray
        # Y_sample = new_Y[:,0].pop(sample) # Ground Truth
        new_X = np.delete(X, sample) # No need to copy since the operations doesn't affect directly X
        new_Y = np.delete(Y, sample) # Same for Y

        # Fitting and predicting
        model.fit(new_X, new_Y)
        #prediction = model.predict(X_sample)
        prediction = model.predict(X[sample])

        # Calculating error and variance of each CV
        #error = (prediction - Y_sample) ** 2
        error = (prediction - Y[sample]) ** 2 # Distance between Prediction and Ground Truth
        variance = model.mle_variance

        # Adding the variance and error of each fold/loop
        test_errors.append(error) # Error
        mle_variances.append(variance) # Variance

    # Calculating the mean of our errors and variances
    average_test_error = np.mean(test_errors)
    average_mle_variance = np.mean(mle_variances)

    #print('avg_error \n', average_test_error)
    #print('avg_variance \n', average_mle_variance)

    return average_test_error, average_mle_variance



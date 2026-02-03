import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from gaussianprocess import GaussianProcess
from standardize import standardize_targets, unstandardize_targets
from crossvalidate import crossvalidate
from negative_log_predictive_density import negative_log_predictive_density
import pickle

def main():
    # data = np.loadtxt("GPs/given.csv", delimiter=",")
    data = np.loadtxt("given.csv", delimiter=",")
    xTr = data[:, :-1].T  #training data should have dimensions (d,n)
    yTr = data[:, -1].reshape(1,-1) #targets should have dimensions (1,n)

    #define hyperparameter grids — you should experiment beyond these default values,
    #including different kernels and kernel parameter and noise values;
    #for numerical stability purposes, do not set the noise less than 1e-6

    noise_vars = [1e-4, 1e-3, 0.01, 0.1, 1]
    paras_dict = {
        'linear': [],  #the linear kernel does not have a hyperparameter; you should update the list accordingly
        'polynomial': [2,3,4,5,6,7,8,9],  #the degree of the polynomial
        'rbf': [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10]  #inverse scale length
    }
    
    best_kernel = None
    best_noise = None
    bestP = None
    lowest_error = float('inf')
    
    #perform x-fold cross-validation (choose # of folds in your implementation)
    results = {}
    for ktype, paras in paras_dict.items():
        best_noise_k, bestP_k, lowest_error_k, error_grid = crossvalidate(xTr, yTr, ktype, noise_vars, paras)
        results[ktype] = (best_noise_k, bestP_k, lowest_error_k, error_grid)
        
        if lowest_error_k < lowest_error:
            best_kernel, best_noise, bestP, lowest_error = ktype, best_noise_k, bestP_k, lowest_error_k
    
    print("Best Kernel:", best_kernel)
    print("Best Noise Variance:", best_noise)
    print("Best Kernel Parameter:", bestP)
    print("Lowest Average x-Fold Cross-Validation Root Mean Squared Error:", np.sqrt(lowest_error))

    #combine best parameters into a dictionary
    best_parameters = {
        'kernel': best_kernel,
        'kpar': bestP,
        'noise': best_noise
    }

    #save the best parameters — MAKE SURE TO INCLUDE THIS IN YOUR SUBMISSION
    pickle.dump(best_parameters, open('best_parameters.pickle', 'wb'))

    # YOUR CODE HERE

    X_train = xTr  # shape (d, n)
    y_train = yTr  # shape (1, n)

    # Standardize targets
    y_train_std, mean, std = standardize_targets(y_train)

    # Train GP using best parameters
    model = GaussianProcess(X_train, y_train_std, best_kernel, bestP, best_noise)

    # Predict on training set (to simulate test)
    y_pred_std, var_pred = model.predict(xTr)  # xTr: shape (d, n)

    # Evaluate NLPD on standardized targets
    nlpd_value = negative_log_predictive_density(y_train_std, y_pred_std, var_pred)
    print("Negative Log Predictive Density (NLPD) on Training Data:", nlpd_value)

    # Unstandardize and compute RMSE
    y_pred_unstd = unstandardize_targets(y_pred_std, mean, std)
    rmse = np.sqrt(mean_squared_error(y_train.flatten(), y_pred_unstd))
    print("Final RMSE (Unstandardized):", rmse)

    # Visualization
    # plt.figure(figsize=(10, 5))
    # plt.plot(y_train, label="True y")
    # plt.plot(y_pred_unstd, label="Predicted y")
    # plt.fill_between(np.arange(len(y_pred_unstd)),
    #                  y_pred_unstd - 2 * np.sqrt(np.diag(var_pred)) * std,
    #                  y_pred_unstd + 2 * np.sqrt(np.diag(var_pred)) * std,
    #                  alpha=0.2, label="Confidence Interval")
    # plt.legend()
    # plt.title("GP Predictions with Uncertainty (Training Data)")
    # plt.xlabel("Data Index")
    # plt.ylabel("House Price ($K)")
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    #Things you may want to implement in main:
    # - creating plots to visualize the accuracy of your predictions
    # - compute NLPD for your standardized training points to make sure your function is reasonable; your
    #   function will be tested using the (hidden) test set

if __name__ == "__main__":
    main()

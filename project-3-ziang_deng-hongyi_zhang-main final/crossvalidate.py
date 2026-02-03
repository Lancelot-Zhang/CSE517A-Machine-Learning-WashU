import numpy as np
import math
from gaussianprocess import GaussianProcess
from sklearn.model_selection import KFold
from standardize import standardize_targets, unstandardize_targets

def crossvalidate(xTr, yTr, ktype, noise_vars, paras):
    """
    INPUT:	
      xTr : dxn input matrix
      yTr : 1xn input continuous target
      ktype : kernel type: 'linear', 'rbf', or 'polynomial'
      noise_vars : list of noise variance values to try
      paras: list of kernel parameters to try
      
    Output:
      best_noise: best performing noise variance
      bestP: best performing kernel parameter
      lowest_error: best performing validation error (mean squared error)
      errors: a matrix where errors[i,j] is the validation error with parameters paras[i] and noise_vars[j]
      
    Trains a GP regressor for all combinations of noise_vars and paras and identifies the best setting.
    """

    # YOUR CODE HERE
    
    # Standardize the target values (zero mean, unit variance)
    yTr_std, mean, std = standardize_targets(yTr)
    
    # Use 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Matrix to store validation errors for each (param, noise) pair
    errors = np.zeros((len(paras), len(noise_vars)))
    
    # Initialize best results
    best_noise = None
    bestP = None
    lowest_error = float('inf')
    
    # Grid search over kernel parameters and noise variances
    for i, param in enumerate(paras):
        for j, noise in enumerate(noise_vars):
            fold_errors = []
            for train_index, val_index in kf.split(xTr):
                # Split columns (since xTr is d×n and yTr is 1×n)
                X_train = xTr[:, train_index]  # shape: d×train_size
                X_val = xTr[:, val_index]      # shape: d×val_size
                y_train = yTr_std[:, train_index]  # shape: 1×train_size
                y_val = yTr_std[:, val_index]      # shape: 1×val_size
                
                # Train GP model
                model = GaussianProcess(X_train, y_train, ktype, param, noise)
                
                # Predict on validation set
                y_pred, pred_vars = model.predict(X_val)
                
                # Compute mean squared error
                y_pred_unstd = unstandardize_targets(y_pred, mean, std)
                y_val_unstd = unstandardize_targets(y_val, mean, std)

                # Calculate RMSE for the fold
                fold_error = np.mean((y_pred_unstd - y_val_unstd) ** 2)
                fold_errors.append(fold_error)
            
            # Average error across all folds
            avg_error = np.mean(fold_errors)
            errors[i, j] = avg_error
            
            # Update best parameters if current error is lower
            if avg_error < lowest_error:
                lowest_error = avg_error
                bestP = param
                best_noise = noise
    
    return best_noise, bestP, lowest_error, errors

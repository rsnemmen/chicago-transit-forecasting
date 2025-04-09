"""
TRAINING
===============
"""
import os, sys
import numpy as np
from pathlib import Path

# For ASCII plots of training progress
import asciichartpy
from IPython.display import clear_output

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import pandas as pd

# For univariate TS
def fit_and_evaluate(model, train_loader, valid_loader, learning_rate, epochs=500, verbose=0):
    # Device configuration (GPU if available)
    device = next(model.parameters()).device

    # Setup optimizer (equivalent to SGD with momentum in TensorFlow)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Loss functions
    huber_loss = torch.nn.HuberLoss()
    mae_loss = torch.nn.L1Loss()

    # Early stopping setup
    patience = 50
    best_val_mae = float('inf')
    patience_counter = 0
    best_model_state = None

    loss_list=[]

    for epoch in range(epochs):
        # Training phase
        model.train()
        for batch_x, batch_y in train_loader:
            # Reshape input: [batch_size, seq_length] -> [batch_size, seq_length, 1]
            batch_x = batch_x.unsqueeze(-1).to(device).float()
            batch_y = batch_y.to(device).float()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_x)

            # Ensure compatible shapes - squeeze the output if needed
            if len(outputs.shape) > len(batch_y.shape):
                outputs = outputs.squeeze()

            # Calculate loss
            loss = huber_loss(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_mae_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                # Reshape input: [batch_size, seq_length] -> [batch_size, seq_length, 1]
                batch_x = batch_x.unsqueeze(-1).to(device).float()
                batch_y = batch_y.to(device).float()

                outputs = model(batch_x)

                # Ensure compatible shapes - squeeze the output if needed
                if len(outputs.shape) > len(batch_y.shape):
                    outputs = outputs.squeeze()

                batch_mae = mae_loss(outputs, batch_y)

                val_mae_sum += batch_mae.item() * batch_x.size(0)
                val_count += batch_x.size(0)

        val_mae = val_mae_sum / val_count
        loss_list.append(np.log10(val_mae)) # for ASCII plot

        # Print progress if requested
        if verbose > 0:
            clear_output(wait=True)
            print(f'Epoch {epoch+1}/{epochs}, Validation MAE: {val_mae:.6f}')
            print(asciichartpy.plot(loss_list[-100:], {'height': 10, 'max':np.max(loss_list)}))


        # Early stopping check
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose > 0:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                break

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_mae * 1e6





# For multivariate TS
def fit_and_evaluate_mulvar(model, train_loader, valid_loader, learning_rate, epochs=500, verbose=0):
    # Device configuration (GPU if available)
    device = next(model.parameters()).device

    # Setup optimizer (equivalent to SGD with momentum in TensorFlow)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Loss functions
    huber_loss = torch.nn.HuberLoss()
    mae_loss = torch.nn.L1Loss()

    # Early stopping setup
    patience = 50
    best_val_mae = float('inf')
    patience_counter = 0
    best_model_state = None

    loss_list=[]

    for epoch in range(epochs):
        # Training phase
        model.train()
        for batch_x, batch_y in train_loader:
            # Handle multivariate input - no need to unsqueeze if already multivariate
            # batch_x shape should be [batch_size, seq_length, num_features]
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_x)

            # Ensure compatible shapes for loss calculation
            # If predicting a single variable, ensure output shape matches target
            if len(batch_y.shape) == 1 and len(outputs.shape) > 1:  # Single target in a batch
                outputs = outputs.squeeze(-1)

            # Calculate loss
            loss = huber_loss(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_mae_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                # Handle multivariate input
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()

                outputs = model(batch_x)

                # Ensure compatible shapes for loss calculation
                if len(batch_y.shape) == 1 and len(outputs.shape) > 1:  # Single target in a batch
                    outputs = outputs.squeeze(-1)

                batch_mae = mae_loss(outputs, batch_y)

                val_mae_sum += batch_mae.item() * batch_x.size(0)
                val_count += batch_x.size(0)

        val_mae = val_mae_sum / val_count
        loss_list.append(np.log10(val_mae)) # for ASCII plot

        # Print progress if requested
        if verbose > 0:
            clear_output(wait=True)
            print(f'Epoch {epoch+1}/{epochs}, Validation MAE: {val_mae:.6f}')
            print(asciichartpy.plot(loss_list[-100:], {'height': 10, 'max':np.max(loss_list)}))


        # Early stopping check
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose > 0:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                break

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_mae * 1e6



# Trains a model and saves weights to disk. If weight file is found, load it from disk.
def train_save(model, in_colab, filename, train_loader, valid_loader, learning_rate, training_fn):
  if in_colab:
    model_file='/content/drive/MyDrive/Colab Notebooks/pytorch/models/'+filename
  else:
    model_file=Path() / "models" / filename

  if not os.path.exists(model_file):
    # train and evaluate
    training_fn(model, train_loader, valid_loader, learning_rate=learning_rate, verbose=1)

    # Save weights to Google Drive
    print("Saving weights to disk")
    torch.save(model.state_dict(), model_file)
  else:
    # Then load the saved weights
    print("Loading weights from disk")
    model.load_state_dict(torch.load(model_file))

    # Set to evaluation mode if using for inference
    model.eval()


"""
INFERENCE
============
"""

# Helper function for plotting a model forecast.
# Single variate.
def forecast(seq_length, ts, model):
    """
    PyTorch version of forecasting function

    Args:
        seq_length: length of input sequence to model
        ts: input pandas time series
        model: trained PyTorch model

    Returns: forecast as pandas Series
    """
    # Set model to evaluation mode
    model.eval()

    # Get device the model is on
    device = next(model.parameters()).device

    # Convert time series to numpy array
    ts_np = ts.to_numpy()

    y_preds = []
    t = []

    # Initialize the sliding window
    t_in = ts_np[:seq_length]

    with torch.no_grad():  # No need to track gradients for inference
        for i, today in enumerate(ts_np[seq_length+1:]):
            # Reshape input for PyTorch model: [batch_size=1, seq_length, features=1]
            t_in_torch = torch.tensor(t_in, device=device).float().unsqueeze(0).unsqueeze(-1)

            # Get prediction
            y_pred = model(t_in_torch)

            # Extract the scalar prediction value
            if hasattr(y_pred, 'squeeze'):
                y_pred = y_pred.squeeze().item()
            else:
                y_pred = y_pred.item()

            # Store prediction and timestamp
            y_preds.append(y_pred)
            t.append(ts.index[seq_length+1+i])

            # Update sliding window (shift left by 1 and add new prediction)
            t_in = np.roll(t_in, -1)
            t_in[-1] = y_pred

    # Return as pandas Series
    return pd.Series(y_preds, index=t)


# Multivariate, one step ahead
def forecast_mulvar(seq_length, ts, model, target_cols=None, exog_cols=None):
    """
    PyTorch version of multivariate forecasting function

    Args:
        seq_length: length of input sequence to model
        ts: input pandas DataFrame containing all variables
        model: trained PyTorch multivariate model
        target_cols: list of column names to forecast (default is first column)
        exog_cols: list of exogenous column names that aren't forecasted but used as inputs
                  (if None, uses all columns as both inputs and potential targets)

    Returns: forecast as pandas Series (if single target) or DataFrame (if multiple targets)
    """
    # Set model to evaluation mode
    model.eval()

    # Get device the model is on
    device = next(model.parameters()).device

    # Default to first column if target_cols not specified
    if target_cols is None:
        target_cols = [ts.columns[0]]
    elif isinstance(target_cols, str):
        target_cols = [target_cols]

    # Default to using all columns if exog_cols not specified
    if exog_cols is None:
        feature_cols = ts.columns.tolist()
    else:
        feature_cols = list(set(target_cols + exog_cols))

    # Number of features in input
    n_features = len(feature_cols)
    # Number of target variables
    n_targets = len(target_cols)

    # Convert time series to numpy array
    ts_np = ts[feature_cols].to_numpy()

    # Initialize storage for predictions
    y_preds = [[] for _ in range(n_targets)]
    t = []

    # Initialize the sliding window with first seq_length observations
    t_in = ts_np[:seq_length].copy()

    with torch.no_grad():  # No need to track gradients for inference
        for i, today in enumerate(ts_np[seq_length+1:]):
            # Reshape input for PyTorch model: [batch_size=1, seq_length, n_features]
            t_in_torch = torch.tensor(t_in, device=device).float().unsqueeze(0)

            # Get prediction
            y_pred = model(t_in_torch)

            # Process prediction based on model output shape
            if n_targets == 1:
                # Single target case
                y_pred_value = y_pred.squeeze().item()
                y_preds[0].append(y_pred_value)
            else:
                # Multiple target case
                y_pred_values = y_pred.squeeze().cpu().numpy()
                for j in range(n_targets):
                    y_preds[j].append(y_pred_values[j])

            # Store timestamp
            t.append(ts.index[seq_length+1+i])

            # Update sliding window (shift left by 1 and add new values)
            t_in = np.roll(t_in, -1, axis=0)

            # For each feature, update the last value
            for j, col in enumerate(feature_cols):
                if col in target_cols:
                    # If this is a target column, use our prediction
                    target_idx = target_cols.index(col)
                    t_in[-1, j] = y_preds[target_idx][-1]
                else:
                    # For exogenous variables, use actual values if available
                    # Otherwise, keep the previous value (assume persistence)
                    if i + seq_length + 1 < len(ts_np):
                        t_in[-1, j] = ts_np[i + seq_length + 1, j]

    # Return as pandas Series (single target) or DataFrame (multiple targets)
    if n_targets == 1:
        return pd.Series(y_preds[0], index=t, name=target_cols[0])
    else:
        return pd.DataFrame({col: vals for col, vals in zip(target_cols, y_preds)}, index=t)



    # Multi-step forecasting
def forecast_ahead(seq_length, data, model, return_numpy=True, device=None):
    """
    Make forecasts using a PyTorch time series model.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model for time series forecasting
    data : pandas.DataFrame or numpy.ndarray
        Input time series data
    seq_length : int
        Length of the sequence to use for prediction
    return_numpy : bool, default=True
        Whether to return predictions as numpy array
    device : torch.device, default=None
        Device to run the model on. If None, uses CUDA if available
        
    Returns:
    --------
    predictions : numpy.ndarray or torch.Tensor
        Forecasted values
    """
    # Determine device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert input data to numpy if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    # Prepare input tensor with shape [1, seq_length, features]
    X = torch.tensor(data[np.newaxis, :seq_length], dtype=torch.float32).to(device)
    
    # Set model to evaluation mode and move to device
    model = model.to(device)
    model.eval()
    
    # Make prediction without computing gradients
    with torch.no_grad():
        predictions = model(X)
    
    # Return as numpy array if requested
    if return_numpy:
        return predictions.cpu().numpy()
    
    return predictions



def forecast_errors(seq_length, data, model, return_numpy=True, device=None, n_samples=100):
    """
    Make forecasts using a PyTorch time series model with optional MC dropout for uncertainty quantification.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model for time series forecasting.
    data : pandas.DataFrame or numpy.ndarray
        Input time series data.
    seq_length : int
        Length of the sequence to use for prediction.
    return_numpy : bool, default=True
        Whether to return predictions as a numpy array.
    device : torch.device, default=None
        Device to run the model on. If None, uses CUDA if available.
    n_samples : int, default=100
        Number of stochastic forward passes for MC dropout. If 1, standard inference is performed.
        
    Returns:
    --------
    If n_samples > 1:
        forecast_mean : numpy.ndarray or torch.Tensor
            Forecasted values (mean over MC dropout samples).
        forecast_std : numpy.ndarray or torch.Tensor
            Standard deviation of the forecasts (uncertainty estimation).
    Else:
        predictions : numpy.ndarray or torch.Tensor
            Forecasted values.
    """
    # Determine device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert input data to numpy if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    # Prepare input tensor with shape [1, seq_length, features]
    X = torch.tensor(data[np.newaxis, :seq_length], dtype=torch.float32).to(device)
    
    # Move model to selected device
    model = model.to(device)
    
    # If we want to perform MC dropout inference (more than one sample), we need to enable dropout at inference.
    # Warning: This sets the entire model in train() mode, so if you have layers like BatchNorm they will also be in training mode.
    if n_samples > 1:
        model.train()  
    else:
        model.eval()
        
    predictions_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X)  # typically shape: [1, forecast_steps]
            predictions_list.append(pred.unsqueeze(0))
    
    # Concatenate predictions along the new "sample" dimension.
    # New shape: [n_samples, 1, forecast_steps]
    predictions_all = torch.cat(predictions_list, dim=0)
    
    if n_samples > 1:
        # Compute the mean and standard deviation across the n_samples
        forecast_mean = predictions_all.mean(dim=0)  # shape: [1, forecast_steps]
        forecast_std = predictions_all.std(dim=0)    # shape: [1, forecast_steps]
        
        if return_numpy:
            forecast_mean = forecast_mean.cpu().numpy()
            forecast_std = forecast_std.cpu().numpy()
        return forecast_mean, forecast_std
    else:
        # Standard inference with a single sample; squeeze the first dimension.
        predictions = predictions_all.squeeze(0)
        if return_numpy:
            return predictions.cpu().numpy()
        return predictions

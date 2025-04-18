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
def train_save(model, train_loader, valid_loader, training_fn, in_colab=False, learning_rate=0.05, epochs=500, filename='model.pth', rewrite=False, verbose=1):
    """
    Trains the model using training_fn if the model file doesn't exist or rewrite is True.
    Loads weights from the file if it exists and rewrite is False.
    Saves the model after training if rewrite is True or the file didn't exist initially.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training data.
        valid_loader: DataLoader for validation data.
        training_fn: The function to call for training (e.g., fit_and_evaluate_mulvar).
                     It should accept (model, train_loader, valid_loader, learning_rate, epochs, verbose)
                     and return a metric (e.g., validation MAE).
        in_colab (bool): Flag for Colab environment path handling.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Maximum number of training epochs.
        filename (str): Name of the file to save/load the model state_dict.
        rewrite (bool): If True, always retrain and overwrite the existing file.
                        If False, load from file if it exists, otherwise train and save.
        verbose (int): Verbosity level for the training function.

    Returns:
        The metric returned by the training_fn if training occurred,
        or None if weights were loaded without retraining. Potentially load and evaluate
        on validation set if weights are loaded to return a consistent metric.
        (Current implementation returns metric only if trained).
    """
    if in_colab:
        # Adjust path for Google Drive if needed
        drive_path = '/content/drive/MyDrive/Colab Notebooks/pytorch/models/'
        if not os.path.exists(drive_path):
             os.makedirs(drive_path) # Ensure directory exists
        model_file = os.path.join(drive_path, filename)
    else:
        model_dir = Path() / "models"
        model_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        model_file = model_dir / filename

    metric = None # Initialize metric variable

    # Decide whether to train or load
    if rewrite or not os.path.exists(model_file):
        if rewrite and os.path.exists(model_file):
            print(f"Retraining and overwriting existing model file: {model_file}")
        elif not os.path.exists(model_file):
             print(f"Model file not found. Training model: {model_file}")
        else: # Should not happen based on logic, but good practice
             print(f"Training model: {model_file}")


        # Train the model
        metric = training_fn(model, train_loader, valid_loader,
                             learning_rate=learning_rate, epochs=epochs, verbose=verbose)

        # Save the trained model state
        print(f"Saving trained weights to disk: {model_file}")
        try:
            torch.save(model.state_dict(), model_file)
        except Exception as e:
            print(f"Error saving model: {e}")


    else:
        # Load existing weights
        print(f"Loading weights from disk: {model_file}")
        try:
            # Ensure loading happens on the correct device (e.g., CPU if saved from GPU and no GPU now)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval() # Set model to evaluation mode after loading
            print("Model loaded successfully.")
            # Optional: Evaluate the loaded model to get a metric if needed
            # metric = evaluate_loaded_model(model, valid_loader) # You'd need a separate eval function
        except Exception as e:
            print(f"Error loading model: {e}. Consider retraining.")
            # Optionally trigger training here if loading fails and you want a fallback
            # print("Retraining model due to loading error...")
            # metric = training_fn(model, train_loader, valid_loader,
            #                      learning_rate=learning_rate, epochs=epochs, verbose=verbose)
            # print(f"Saving newly trained weights to disk: {model_file}")
            # torch.save(model.state_dict(), model_file)


    return metric # Return the metric obtained from training (or None/evaluation if loaded)




"""
INFERENCE
============

Rolling Forecast:

The forecast* functions below construct a sliding window (t_in) containing the most recent 
seq_length observations. The trained model’s is called window, and the prediction is stored.
The timestamp for the prediction is taken from the DataFrame’s index.
The sliding window is updated: for target columns, the forecasted value is inserted; 
for exogenous columns, the actual future value is inserted (if available), 
otherwise persistence is assumed.
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




def forecast_mulvar_xgb(seq_length, ts, model, target_cols=None, exog_cols=None):
    """
    XGBoost version of the multivariate forecasting function for one-step-ahead forecasts.

    Args:
        seq_length (int): Length of input sequence to the model (e.g., 56 days).
        ts (pd.DataFrame): Input DataFrame with all variables and a datetime index.
        model: Trained XGBoost model (e.g., XGBRegressor or a multi-output wrapper) that outputs 
               the forecast for the target variable(s).
        target_cols: List of column names to forecast (default is the first column).
        exog_cols: List of exogenous column names that are used as inputs, but not forecasted.
                  If None, all columns are used as both inputs and candidates for forecasting.

    Returns:
        pd.Series or pd.DataFrame: Forecast results (Series for a single target, DataFrame for multiple).
    """
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
    
    n_features = len(feature_cols)
    n_targets = len(target_cols)
    
    # Convert DataFrame to a numpy array (using only the selected feature columns)
    ts_np = ts[feature_cols].to_numpy()
    
    # Initialize storage for predictions. One list per target
    y_preds = [[] for _ in range(n_targets)]
    forecast_timestamps = []

    # Initialize the sliding window using the first seq_length rows.
    # The window has shape (seq_length, n_features)
    t_in = ts_np[:seq_length].copy()

    # Loop over the available time steps (simulate rolling forecast)
    # We start at seq_length+1 so that the forecast is for t+1 relative to the current window.
    for i, _ in enumerate(ts_np[seq_length+1:]):
        # Flatten the current window into shape (1, seq_length * n_features)
        t_in_flat = t_in.flatten().reshape(1, -1)
        
        # Get the forecast from the XGBoost model
        y_pred = model.predict(t_in_flat)
        
        # Process prediction output. If only one target is being forecasted, then y_pred is
        # a 1-D array with a single value. For multiple targets, we assume y_pred returns an array
        # of shape (n_targets, ) or (1, n_targets).
        if n_targets == 1:
            y_pred_value = y_pred[0]
            y_preds[0].append(y_pred_value)
        else:
            # Ensure we have a flat array of predictions
            y_pred_values = y_pred.flatten()
            for j in range(n_targets):
                y_preds[j].append(y_pred_values[j])
        
        # Store the timestamp corresponding to the forecasted observation.
        forecast_timestamps.append(ts.index[seq_length+1 + i])
        
        # Update the sliding window:
        # 1. Shift the window one time step ahead.
        t_in = np.roll(t_in, -1, axis=0)
        
        # 2. For the most recent row, put the forecasted values for target columns and actual values
        # for exogenous columns.
        for j, col in enumerate(feature_cols):
            if col in target_cols:
                # For target columns, use the forecast
                target_idx = target_cols.index(col)
                t_in[-1, j] = y_preds[target_idx][-1]
            else:
                # For exogenous variables, if available, use the actual following value
                if i + seq_length + 1 < len(ts_np):
                    t_in[-1, j] = ts_np[i + seq_length + 1, j]

    # Return forecasted results as a pandas Series (for a single target) or DataFrame (multiple targets)
    if n_targets == 1:
        return pd.Series(y_preds[0], index=forecast_timestamps, name=target_cols[0])
    else:
        forecast_df = pd.DataFrame({col: vals for col, vals in zip(target_cols, y_preds)},
                                   index=forecast_timestamps)
        return forecast_df





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



# MISC
# =========

# Computes MAE for two numpy arrays
def maef(y_pred, data):
    return (y_pred - data).abs().mean()

# Align the phase: helper function
def align_phase(ts, dt=-1):
  """
  Returns a pandas time series object.
  """
  # shift the TS by -1 day
  values_shift=np.roll(ts, dt)

  # create a pandas TS
  return pd.Series(values_shift, index=ts.index)
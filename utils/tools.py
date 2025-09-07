import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils.constants import EXCLUDE_COLS
import os
import glob
from tqdm import tqdm
import joblib

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def compute_scaling_parameters(data_path, save_path='scaler.joblib', exclude_cols=EXCLUDE_COLS):
    """
    Scan every *.csv.gz file in data_path, fit an incremental StandardScaler
    on all sensor columns EXCEPT those listed in EXCLUDE_COLS, then save it.
    """
    print("Computing scaling parameters from all files…")
    file_paths = sorted(glob.glob(os.path.join(data_path, "WTG_*_time_chunk_*.csv.gz")))
    print(f"Found {len(file_paths)} files in {data_path}")

    scaler = StandardScaler()
    total_samples = 0
    first_file_columns = None
    
    for i, fp in enumerate(tqdm(file_paths, desc="Processing files")):
        df = pd.read_csv(fp, compression='gzip')
        
        # get sensor columns for this file
        sensor_cols = df.columns.difference(exclude_cols)
        
        # Track columns from first file for comparison
        if i == 0:
            first_file_columns = set(sensor_cols)
            print(f"First file has {len(sensor_cols)} features: {sorted(sensor_cols)}")
        
        # Check for column differences compared to first file
        if set(sensor_cols) != first_file_columns:
            print(f"\nFILE MISMATCH DETECTED in {os.path.basename(fp)}")
            print(f"This file has {len(sensor_cols)} features (expected {len(first_file_columns)})")
            
            # Find extra columns
            extra_cols = set(sensor_cols) - first_file_columns
            if extra_cols:
                print(f"Extra columns: {sorted(extra_cols)}")
            
            # Find missing columns
            missing_cols = first_file_columns - set(sensor_cols)
            if missing_cols:
                print(f"Missing columns: {sorted(missing_cols)}")
                
        # Continue with normal processing
        valid_rows = df.loc[~df['missing_data'], sensor_cols].values
        
        # Print shapes for debugging
        print(f"\nFile {i+1}/{len(file_paths)}: {os.path.basename(fp)}")
        print(f"Sensor columns: {len(sensor_cols)}")
        print(f"Valid rows shape: {valid_rows.shape}")
        
        if len(valid_rows):
            try:
                if total_samples > 0:
                    print(f"Scaler expects {len(scaler.mean_)} features")
                
                scaler.partial_fit(valid_rows)
                total_samples += len(valid_rows)
                print(f"Success - scaler now has {len(scaler.mean_)} features")
                
            except ValueError as e:
                print(f"ERROR in file {os.path.basename(fp)}")
                print(f"Error message: {str(e)}")
                print(f"Scaler expected {len(scaler.mean_)} features but got {valid_rows.shape[1]}")
                
                # Print full column comparison
                if hasattr(scaler, 'feature_names_in_'):
                    prev_cols = set(scaler.feature_names_in_)
                    curr_cols = set(sensor_cols)
                    
                    extra = curr_cols - prev_cols
                    missing = prev_cols - curr_cols
                    
                    print(f"Extra columns in this file: {sorted(extra)}")
                    print(f"Missing columns in this file: {sorted(missing)}")
                
                raise  # Re-raise to stop processing
    
    print(f"Scaler fit on {total_samples} rows | {len(scaler.mean_)} features")
    joblib.dump(scaler, save_path)
    print(f"Scaler saved to {save_path}")
    return scaler







# class StandardScaler():
#     def __init__(self):
#         self.mean = 0.
#         self.std = 1.
    
#     def fit(self, data):
#         self.mean = data.mean(0)
#         self.std = data.std(0)

#     def transform(self, data):
#         mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
#         std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
#         return (data - mean) / std

#     def inverse_transform(self, data):
#         mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
#         std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
#         if data.shape[-1] != mean.shape[-1]:
#             mean = mean[-1:]
#             std = std[-1:]
#         return (data * std) + mean
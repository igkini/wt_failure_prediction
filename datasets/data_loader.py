import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')
import glob
import re
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List, Tuple, Dict
from utils.constants import EXCLUDE_COLS, FAILURE_PERIODS, PREDICTION_WINDOWS, LABEL_LOOKAHEAD
from datasets.utils import fill_window, code_features, window_overlaps_failure, window_overlaps_ambiguous, window_overlaps_outlier


class WindTurbinePredictionWindowDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        scaler_path: str = "./wind_turbine_scaler.joblib",
        window_size: int = 24,
        stride: int = 12,
        timeenc: int = 1,
        freq: str = "h",
        max_codes_per_timestep: int = 10, 
        max_missing_rows: int = 6*8, 
    ) -> None:
        super().__init__()

        self.root_path = root_path
        self.seq_len = window_size
        self.stride = stride
        self.timeenc = timeenc
        self.freq = freq
        self.max_codes_per_timestep = max_codes_per_timestep
        self.max_missing_rows = max_missing_rows

        if scaler_path and os.path.exists(scaler_path):
            self.scaler: StandardScaler = joblib.load(scaler_path)
            self.scaling_enabled = True
        else:
            # identity scaler – values remain unchanged
            self.scaler = StandardScaler()
            self.scaler.mean_ = 0.0
            self.scaler.scale_ = 1.0
            self.scaling_enabled = False
            print("⚠  Scaler not found – no scaling will be applied.")

        self.file_paths: List[str] = sorted(
            glob.glob(os.path.join(root_path, "WTG_*_time_chunk_*.csv.gz"))
        )
        print(f"Found {len(self.file_paths)} files in {root_path}")

        self.data_x: List[np.ndarray] = []
        self.data_time: List[np.ndarray] = []
        self.data_code: List[np.ndarray] = []

        self.window_start_dates: List[pd.Timestamp] = []
        self.window_end_dates: List[pd.Timestamp] = []
        self.window_turbine_ids: List[str] = []

        self.__read_data__()

    def _horizon_in_target_range(self, horizon_ts: pd.Timestamp) -> bool:
        return any(start <= horizon_ts <= end for start, end in PREDICTION_WINDOWS)

    def __read_data__(self) -> None:
        print("Preparing data…")
        dropped_outside_target = 0
        dropped_too_many_missing = 0

        for fp in tqdm(self.file_paths, desc="Files"):
            m = re.search(r"WTG_(\d+)", fp)
            if not m:
                continue
            wtg_key = f"WT{int(m.group(1))}"

            df = pd.read_csv(fp, compression="gzip")
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])

            # separate sensor block and codes
            sensor_cols = df.columns.difference( EXCLUDE_COLS )

            rows = df.set_index("Timestamp")
            sensor_df = rows[sensor_cols]
            code_series = rows["all_codes"]

            # sliding-window extraction
            for start in range(0, len(sensor_df) - self.seq_len + 1, self.stride):
                end = start + self.seq_len
                win_start_ts = sensor_df.index[start]
                win_end_ts = sensor_df.index[end - 1]

                window_df = sensor_df.iloc[start:end]
                missing_rows = window_df.isna().all(axis=1).sum()
                if missing_rows > self.max_missing_rows:
                    dropped_too_many_missing += 1
                    continue

                # keep only if 10-day horizon is in PREDICTION_WINDOWS
                horizon_ts = win_end_ts +  LABEL_LOOKAHEAD
                if not self._horizon_in_target_range(horizon_ts):
                    dropped_outside_target += 1
                    continue

                # scale this specific window
                if self.scaling_enabled:
                    window_df_scaled = pd.DataFrame(
                        self.scaler.transform(window_df),
                        index=window_df.index,
                        columns=window_df.columns
                    )
                else:
                    window_df_scaled = window_df

                # fill missing values inside scaled window
                slice_arr = fill_window(window_df_scaled, 0, len(window_df_scaled))

                # code features
                code_window = code_series.iloc[start:end]
                code_arr = code_features(code_window, self.max_codes_per_timestep)

                # time-encoding
                ts_slice = sensor_df.index[start:end]
                stamp = time_features(
                    pd.DataFrame({"date": ts_slice}),
                    timeenc=self.timeenc,
                    freq=self.freq,
                )

                # store
                self.data_x.append(slice_arr)
                self.data_time.append(stamp)
                self.data_code.append(code_arr)

                self.window_start_dates.append(win_start_ts)
                self.window_end_dates.append(win_end_ts)
                self.window_turbine_ids.append(wtg_key)

        print("\n=== Prediction-window Dataset Summary ===")
        print(f"Windows kept                     : {len(self.data_x):>6}")
        print(f"Dropped (horizon outside range)  : {dropped_outside_target:>6}")
        print(f"Dropped (>max missing threshold) : {dropped_too_many_missing:>6}")
        print(f"Max missing rows threshold       : {self.max_missing_rows:>6}")
        print(f"Max codes per timestep           : {self.max_codes_per_timestep:>6}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = torch.FloatTensor(self.data_x[idx])
        x_time = torch.FloatTensor(self.data_time[idx])
        x_code = torch.tensor(self.data_code[idx], dtype=torch.long)

        return x, x_time, x_code

    def __len__(self) -> int:
        return len(self.data_x)

    def get_window_dates(self, idx: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return self.window_start_dates[idx], self.window_end_dates[idx]

    def get_window_info(self, idx: int) -> Dict:
        return {
            "start_date": self.window_start_dates[idx],
            "end_date": self.window_end_dates[idx],
            "turbine_id": self.window_turbine_ids[idx],
        }

class WindTurbineFailureDatasetMultiCode(Dataset):
    def __init__(
        self,
        root_path: str,
        scaler_path: str = "./wind_turbine_scaler.joblib",
        window_size: int = 24,
        stride: int = 12,
        timeenc: int = 1,
        freq: str = "h",
        
        max_codes_per_timestep: int = 5,
        max_missing_rows: int = 24*2, 
    ) -> None:
        super().__init__()

        self.root_path = root_path
        self.seq_len = window_size
        self.stride = stride
        self.timeenc = timeenc
        self.freq = freq
        self.max_codes_per_timestep = max_codes_per_timestep
        self.max_missing_rows = max_missing_rows  


        if scaler_path and os.path.exists(scaler_path):
            self.scaler: StandardScaler = joblib.load(scaler_path)
            self.scaling_enabled = True
        else:
            self.scaler = StandardScaler()
            self.scaler.mean_ = 0.0
            self.scaler.scale_ = 1.0
            self.scaling_enabled = False
            print("⚠  Scaler not found – no scaling will be applied.")


        self.file_paths: List[str] = sorted(
            glob.glob(os.path.join(root_path, "WTG_*_time_chunk_*.csv.gz"))
        )
        print(f"Found {len(self.file_paths)} files in {root_path}")

        self.data_x: List[np.ndarray] = [] 
        self.data_time: List[np.ndarray] = []
        self.data_code: List[np.ndarray] = []  
        self.labels: List[int] = []            
        
        self.window_start_dates: List[pd.Timestamp] = []
        self.window_end_dates: List[pd.Timestamp] = []
        self.window_turbine_ids: List[str] = []
        
        self.__read_data__()

    def __read_data__(self) -> None:
        print("Preparing data...")
        dropped_overlap_failure = 0
        dropped_overlap_ambiguous = 0
        dropped_overlap_outlier = 0
        dropped_too_many_missing = 0

        for fp in tqdm(self.file_paths, desc="Files"):
            m = re.search(r"WTG_(\d+)", fp)
            if not m:
                continue
            wtg_key = f"WT{int(m.group(1))}"

            df = pd.read_csv(fp, compression="gzip")
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])

            # separate sensor block and codes
            sensor_cols = df.columns.difference(EXCLUDE_COLS)

            rows = df.set_index("Timestamp")
            sensor_df = rows[sensor_cols]
            code_series = rows["all_codes"]

            # sliding‑window extraction (no pre-scaling of entire dataframe)
            for start in range(0, len(sensor_df) - self.seq_len + 1, self.stride):
                end = start + self.seq_len
                window_start_ts = sensor_df.index[start]
                window_end_ts = sensor_df.index[end - 1]
                
                # check if window overlaps with failure period
                if window_overlaps_failure(window_start_ts, window_end_ts, wtg_key):
                    dropped_overlap_failure += 1
                    continue
            
                # check if window overlaps with ambiguous period minus 10 days
                if window_overlaps_ambiguous(window_start_ts, window_end_ts, wtg_key):
                    dropped_overlap_ambiguous += 1
                    continue
            
                # check if window overlaps with outlier period minus 10 days
                if window_overlaps_outlier(window_start_ts, window_end_ts, wtg_key):
                    dropped_overlap_outlier += 1
                    continue
            
                # extract window and check missing data threshold
                window_df = sensor_df.iloc[start:end]
                missing_rows = window_df.isna().all(axis=1).sum()
                if missing_rows > self.max_missing_rows:
                    dropped_too_many_missing += 1
                    continue
            
                # scale this specific window (like in prediction dataset)
                if self.scaling_enabled:
                    window_df_scaled = pd.DataFrame(
                        self.scaler.transform(window_df),
                        index=window_df.index,
                        columns=window_df.columns
                    )
                else:
                    window_df_scaled = window_df

                # fill missing values inside scaled window
                slice_arr = fill_window(window_df_scaled, 0, len(window_df_scaled))
            
                # check if notification in horizon
                last_ts = window_end_ts
                horizon_ts = last_ts + LABEL_LOOKAHEAD

                notif_in_horizon = any(
                    last_ts < s <= horizon_ts for s, _ in FAILURE_PERIODS.get(wtg_key, [])
                )
            
                label = 1 if notif_in_horizon else 0
                
                # code features
                code_window = code_series.iloc[start:end]
                code_arr = code_features(code_window, self.max_codes_per_timestep)
            
                # time-encoding
                ts_slice = sensor_df.index[start:end]
                stamp = time_features(
                    pd.DataFrame({"date": ts_slice}), timeenc=self.timeenc, freq=self.freq
                )
            
                # store
                self.data_x.append(slice_arr)
                self.data_time.append(stamp)
                self.data_code.append(code_arr)
                self.labels.append(label)
                self.window_start_dates.append(window_start_ts)
                self.window_end_dates.append(window_end_ts)
                self.window_turbine_ids.append(wtg_key)

        print("\n=== Dataset Summary (Multi-Code Support with Interpolation) ===")
        print(f"Windows kept                     : {len(self.data_x):>6}")
        print(f"Dropped (overlap failure)       : {dropped_overlap_failure:>6}")
        print(f"Dropped (overlap ambiguous)     : {dropped_overlap_ambiguous:>6}")
        print(f"Dropped (overlap outlier)       : {dropped_overlap_outlier:>6}")
        print(f"Dropped (>max missing threshold): {dropped_too_many_missing:>6}")
        print(f"Max missing rows threshold      : {self.max_missing_rows:>6}")
        print(f"Max codes per timestep          : {self.max_codes_per_timestep:>6}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        x = torch.FloatTensor(self.data_x[idx])
        x_time = torch.FloatTensor(self.data_time[idx])
        x_code = torch.tensor(self.data_code[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, x_time, x_code, y
    
    def __len__(self) -> int:
        return len(self.data_x)
    
    def get_window_dates(self, idx: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return self.window_start_dates[idx], self.window_end_dates[idx]
    
    def get_window_info(self, idx: int) -> Dict:
        return {
            'start_date': self.window_start_dates[idx],
            'end_date': self.window_end_dates[idx],
            'turbine_id': self.window_turbine_ids[idx],
            'label': self.labels[idx]
        }
    
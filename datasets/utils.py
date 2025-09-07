import pandas as pd
import numpy as np
from utils.constants import  NOTIF_LOOKAHEAD, AMBIGUOUS_PERIODS, OUTLIER_PERIODS, FAILURE_PERIODS

def is_in_failure(ts: pd.Timestamp, wtg_key: str) -> bool:
    """True if *ts* lies inside a *known* failure period for *wtg_key*."""
    return any(s <= ts <= e for s, e in FAILURE_PERIODS.get(wtg_key, []))

def window_overlaps_failure(start_ts: pd.Timestamp, end_ts: pd.Timestamp, wtg_key: str) -> bool:
    """True if the window [start_ts, end_ts] overlaps with any failure period for wtg_key."""
    for s, e in FAILURE_PERIODS.get(wtg_key, []):
        if start_ts <= e and end_ts >= s:
            return True
    return False

def window_overlaps_ambiguous(start_ts: pd.Timestamp, end_ts: pd.Timestamp, wtg_key: str) -> bool:
    """True if the window [start_ts, end_ts] overlaps with any ambiguous period minus 10 days for wtg_key."""
    for s, e in AMBIGUOUS_PERIODS.get(wtg_key, []):
        adjusted_s = s - NOTIF_LOOKAHEAD
        if start_ts <= e and end_ts >= adjusted_s:
            return True
    return False

def window_overlaps_outlier(start_ts: pd.Timestamp, end_ts: pd.Timestamp, wtg_key: str) -> bool:
    """True if the window [start_ts, end_ts] overlaps with any outlier period minus 10 days for wtg_key."""
    for s, e in OUTLIER_PERIODS.get(wtg_key, []):
        adjusted_s = s - NOTIF_LOOKAHEAD
        if start_ts <= e and end_ts >= adjusted_s:
            return True
    return False

def code_features(code_series: pd.Series, max_codes_per_timestep: int) -> np.ndarray:

    code_matrix = []
    
    for code_str in code_series:
        # Parse codes
        if pd.isna(code_str) or code_str == "" or code_str == "0":
            codes = [0]
        else:
            try:
                codes = [int(code.strip()) for code in str(code_str).split('|') if code.strip()]
                codes = codes if codes else [0]
            except (ValueError, AttributeError):
                codes = [0] 
        
        # Pad
        if len(codes) > max_codes_per_timestep:
            padded_codes = codes[:max_codes_per_timestep]
        else:
            padded_codes = codes + [0] * (max_codes_per_timestep - len(codes))
        
        code_matrix.append(padded_codes)
    
    return np.array(code_matrix, dtype=np.int64)

def fill_window( df: pd.DataFrame, start: int, end: int) -> np.ndarray:
        window = df.iloc[start:end].copy()
        window.interpolate(
            method="linear",
            limit_direction="both",
            axis=0,
            inplace=True,
        )
        window.fillna(0.0, inplace=True)
        return window.to_numpy(dtype=np.float32)

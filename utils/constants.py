import pandas as pd
from typing import Dict, List, Tuple

EXCLUDE_COLS = [
    'Gear Bearing Temp. Avg.',
    'Gear Oil Temp. Avg.',
    'Spinner Temp. SlipRing Avg.',

    # 'Generator Bearing2 Temp. Avg.',
    # 'Generator Bearing Temp. Avg.',
    # 'Generator CoolingWater Temp. Avg.',
    # 'Generator Phase1 Temp. Avg.',
    # 'Generator Phase2 Temp. Avg.',
    # 'Generator Phase3 Temp. Avg.',
    # 'Generator RPM Avg.',
    # 'Generator RPM Max.',
    # 'Generator RPM Min.',
    # 'Generator RPM StdDev',

    # Gear / hydraulic system
    
    # 'Gear Oil TemperatureBasis Avg.',
    # 'Hydraulic Oil Temp. Avg.',
    # 'Gear Oil TemperatureLevel1 Avg.',
    # 'Gear Oil TemperatureLevel2_3 Avg.',

    # Rotor
    
    # 'Rotor RPM Avg.',
    # 'Rotor RPM Max.',
    # 'Rotor RPM Min.',
    # 'Rotor RPM StdDev',

    # Ambient conditions
    
    
    'Ambient WindDir Absolute Avg.',
    'Ambient WindDir Relative Avg.',
    'Avg. direction',
    
    # 'Ambient WindSpeed Estimated Avg.',
    # 'Ambient WindSpeed Avg.',
    # 'Ambient WindSpeed Max.',
    # 'Ambient WindSpeed Min.',
    # 'Ambient WindSpeed StdDev',
    # 'Ambient Temp. Avg.',
    
    
    # Power limits / production
    'Active power limit',
    
    # 'Grid Production Power Max.',
    # 'Grid Production Power Min.',
    # 'Grid Production Power StdDev',
    # 'Grid Production Power Avg.',
    
    'Total Active power',
    'Production LatestAverage Active Power Gen 0 Avg.',
    'Production LatestAverage Active Power Gen 1 Avg.',
    'Production LatestAverage Active Power Gen 2 Avg.',

    # Blade & spinner
    'Blades PitchAngle Avg.',
    'Blades PitchAngle Max.',
    'Blades PitchAngle Min.',
    'Blades PitchAngle StdDev',
    
    # 'Spinner Temp. Avg.',

    # Nacelle
    
    # 'Nacelle Temp. Avg.',
    
    'most_sev_code',
    'all_codes',
    "Timestamp",
    "missing_data"
]

NOTIF_LOOKAHEAD = pd.Timedelta(days=20)
LABEL_LOOKAHEAD = pd.Timedelta(days=10)

AMBIGUOUS_PERIODS: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {
    'WT1': [
        (pd.Timestamp("2021-12-09"), pd.Timestamp("2022-01-31")),
        (pd.Timestamp("2023-01-08"), pd.Timestamp("2023-02-27")),
    ],
    'WT2': [
        (pd.Timestamp("2021-12-09"), pd.Timestamp("2022-01-31")),
        (pd.Timestamp("2023-01-08"), pd.Timestamp("2023-02-27")),
        (pd.Timestamp("2024-03-13"), pd.Timestamp("2024-04-18")),
    ],
    'WT3': [
        # (pd.Timestamp("2021-12-09"), pd.Timestamp("2022-01-31")),
        (pd.Timestamp("2023-01-08"), pd.Timestamp("2023-02-27")),
        (pd.Timestamp("2024-03-13"), pd.Timestamp("2024-04-18")),
    ],
    'WT4': [
        (pd.Timestamp("2021-12-09"), pd.Timestamp("2022-01-31")),
        (pd.Timestamp("2023-01-08"), pd.Timestamp("2023-02-27")),
        (pd.Timestamp("2024-03-13"), pd.Timestamp("2024-04-18")),
    ],
    'WT5': [
        (pd.Timestamp("2021-12-09"), pd.Timestamp("2022-01-31")),
        (pd.Timestamp("2023-01-08"), pd.Timestamp("2023-02-27")),
        (pd.Timestamp("2024-03-13"), pd.Timestamp("2024-04-18")),
    ],
    'WT6': [
        (pd.Timestamp("2021-12-09"), pd.Timestamp("2022-01-31")),
        (pd.Timestamp("2023-01-08"), pd.Timestamp("2023-02-27")),
        (pd.Timestamp("2024-03-13"), pd.Timestamp("2024-04-18")),
    ],
    'WT7': [
        (pd.Timestamp("2021-12-09"), pd.Timestamp("2022-01-31")),
        (pd.Timestamp("2023-01-08"), pd.Timestamp("2023-02-27")),
    ],
    'WT8': [
        (pd.Timestamp("2021-12-09"), pd.Timestamp("2022-01-31")),
        (pd.Timestamp("2023-01-08"), pd.Timestamp("2023-02-27")),
        (pd.Timestamp("2024-03-13"), pd.Timestamp("2024-04-18")),
    ],
    'WT9': [
        (pd.Timestamp("2021-12-09"), pd.Timestamp("2022-01-31")),
        (pd.Timestamp("2023-01-08"), pd.Timestamp("2023-02-27")),
        (pd.Timestamp("2024-03-13"), pd.Timestamp("2024-04-18")),
    ],
    'WT10': [
        (pd.Timestamp("2021-12-09"), pd.Timestamp("2022-01-31")),
        (pd.Timestamp("2023-01-08"), pd.Timestamp("2023-02-27")),
        (pd.Timestamp("2024-03-13"), pd.Timestamp("2024-04-18")),
    ],
}

OUTLIER_PERIODS: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {
    'WT5': [
        (pd.Timestamp("2023-11-20"), pd.Timestamp("2024-01-06")),
        (pd.Timestamp("2023-07-07"), pd.Timestamp("2023-07-08")),
    ],
    'WT3': [
        (pd.Timestamp("2023-04-06"), pd.Timestamp("2023-07-03")),
    ],
}

FAILURE_PERIODS: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {
    'WT1': [(pd.Timestamp('2024-01-22'), pd.Timestamp('2024-02-16'))],
    'WT3': [
        (pd.Timestamp('2021-11-09'), pd.Timestamp('2022-01-15')),
        (pd.Timestamp('2020-05-26'), pd.Timestamp('2020-06-09'))
    ],
    'WT5': [(pd.Timestamp('2019-11-25'), pd.Timestamp('2019-12-18'))],
    'WT7': [(pd.Timestamp('2024-02-26'), pd.Timestamp('2024-04-16'))],
}

PREDICTION_WINDOWS = [
    (pd.Timestamp("2021-12-09"), pd.Timestamp("2022-01-31")),
    (pd.Timestamp("2023-01-08"), pd.Timestamp("2023-02-27")),
    (pd.Timestamp("2024-03-13"), pd.Timestamp("2024-04-18")),
]

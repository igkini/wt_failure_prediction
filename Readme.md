# Wind Turbine Failure Prediction

This repository provides an implementation for predicting wind turbine failures using a SCADA dataset.
It includes:
- A custom **PyTorch dataset** implementation (data_loader.py)
- Implementation of a **transformer-based model** (models/)
- **Training and Inference scripts** (train.py and infer.py)

## Overall Approach  

The problem was modeled as a **binary classification task**. A transformer-based model processes sequences of sensor measurements and event logs to predict whether they indicate a **potential failure within the next 10 days**.

## Dataset  

The **raw dataset** consisted of two main components:

1. **SCADA Sensor Data**  
   - Separate CSV files for 10 wind turbines (3 files per turbine).  
   - Contains multiple sensor readings recorded at **10-minute intervals**.  


| Timestamp           | Generator RPM Avg. | Rotor RPM Avg. | Ambient WindSpeed Avg. | Grid Production Power Avg. |
|---------------------|--------------------|----------------|-------------------------|-----------------------------|
| 2023-03-01 00:00:00 | 1221.0             | 11.6           | 6.4                     | 447.0                       |
| 2023-03-01 00:10:00 | 1300.0             | 12.3           | 6.8                     | 540.9                       |
| 2023-03-01 00:20:00 | 1307.0             | 12.4           | 6.9                     | 521.3                       |
| 2023-03-01 00:30:00 | 1331.0             | 12.6           | 6.9                     | 561.0                       |


2. **Event / Log Data**  
   - Provided in separate CSV files.  
   - Contains maintenance and failure logs, including:  
     - Log code ids 
     - Severity levels  
     - Event type (Alarm, Warning, Operation, System Log)  
     - Text descriptions  
   - Recorded at **irregular time intervals**.  


| Detected             | Code | Description                    | Event type        | Severity |
|----------------------|------|--------------------------------|-------------------|----------|
| 2023-03-13 00:25:48  | 265  | Wind: 8.8 m/s  Gen: 1634.1 RPM | Operation log (O) | 1        |
| 2023-03-13 00:25:48  | 266  | Pitch: -2.7  Power: 1125.0 kW  | Operation log (O) | 1        |
| 2023-03-13 00:55:48  | 265  | Wind: 7.3 m/s  Gen: 1438.9 RPM | Operation log (O) | 1        |
| 2023-03-13 00:55:48  | 266  | Pitch: -2.6  Power: 761.2 kW   | Operation log (O) | 1        |


### Data Processing  
- The corresponding **sensor** and **log** CSV files were first **merged** to create unified datasets for each turbine.  
- Each merged file contained:  
  - **Timestamps** at 10-minute intervals  
  - All available **sensor readings**  
  - A **log code column**, where multiple codes occurring within the same 10-minute window were concatenated using `"|"`, sorted by severity.  
- All unique **log codes ids** from the dataset were stored in a **separate CSV file** to build an index map.  
  - This index map is used to create the **vocabulary** required for log code embeddings in the model.  

| Timestamp           | Generator RPM Avg. | Rotor RPM Avg. | Grid Production Power Avg. | all_codes |
|---------------------|--------------------|----------------|-----------------------------|-----------|
| 2023-03-01 00:00:00 | 1221.0             | 11.6           | 447.0                       | 0         |
| 2023-03-01 00:10:00 | 1300.0             | 12.3           | 540.9                       | 0         |
| 2023-03-01 00:20:00 | 1307.0             | 12.4           | 521.3                       | 0         |
| 2023-03-01 00:30:00 | 1331.0             | 12.6           | 561.0                       | 0         |

**Note:** Here, `0` corresponds to **code 1** in the index map. Since no other codes appeared in this sample, only `0` is shown in the `all_codes` column, and no `"|"` separator is needed.  


### PyTorch Dataset 

To allow easier reconfiguration of the dataset, further processing is conducted in the Pytorch dataset implementation(datasets/data_loader.py) rather than at the CSV merging stage.
Key steps implemented in the PyTorch include:

- **Column Filtering**  
  - Keeps only a subset of sensor readings, excluding those defined in `EXCLUDE_COLUMNS`.  

- **Feature Extraction**
  - Time-based features are extracted using the `time_features` function.  
  - Log codes are transformed into categorical features using the `code_features` function, which maps up to `max_codes_per_timestep` log codes per timestep into a list of vocabulary indices.

- **Sliding Window**  
  - Applies a sliding window of length `seq_len` with a configurable `stride` to create sequences of data.  

- **Data Cleaning**
  - Windows overlapping with predefined outlier periods are excluded.
  - Windows with more than `max_missing_rows` missing values are discarded.
  - Missing values in windows with less than `max_missing_rows` missing values are lineary interpolated.

- **Feature Scaling**  
  - A **StandardScaler** is applied to all sensor features.  
  - The **mean** and **standard deviation** are calculated across the entire dataset, and these values are used to scale the data.

 **Note:** The current architecture does not implement **masking mechanisms** for missing values within the model. Missing data is handled only during preprocessing (e.g., linear interpolation).

Separate dataset classes were implemented for **training/validation/testing** and for **inference**.

## Model Architecture

The implementation is an adaptation of the [Informer architecture](https://github.com/zhouhaoyi/Informer2020).  
The architecture was studied in **depth**, and the **transformer-based encoder with the distillation technique** served as the main component of the model.  

## Embeddings
Embeddings were implemented as follows: 

- **Sensor readings** were embedded using a **1D convolutional layer**(`kernel_size=3 and in_channels=num_sensor_cols and out_channels=d_model`).  
- **Positional embeddings** similar to those in the original Transformers paper.
- In the original Informer, the `time_features` function generates multiple time attributes (e.g., month, day, weekday, hour) which are then projected with a linear layer to `d_model`. 
  - In this work, the `time_features` function was customized to include only **month of the year** and **hour of the day**, using a custom frequency `'d'`.  
- **Log ID Embeddings**  
  - Each log code is mapped from the constructed vocabulary to a dense vector using a **learned embedding layer**.  
  - Since the processed input contains a **list of log codes per timestep**, an embedding is computed for each code in the list.  
  - The resulting vectors are then **summed** to form a single log code embedding for that timestep.
- These embeddings are added to form the **final embedding**.

### Encoder

The encoder is implemented as a **stack of encoder layers**. Each encoder layer consists of:

- A **multi-head attention layer** to capture dependencies across the input sequence.  
- Two consecutive **Conv1D blocks** with `kernel_size = 1` (basically Linear layers) that project features to a higher dimension(`d_ff`) and then back to the original dimension(`d_model`).  

If the **distillation option (`distil=True`)** is enabled, then a **Conv1D** layer followed by a **MaxPool** layer reduce the size of the sequence to half(`kernel_size=3`, `stride=2`, `padding=1`).


### Classification head

After the encoder:
- A **pooling layer** to reduce the sequence to `1`, followed by a **Linear layer(1 hidden)** was applied, producing **2 outputs** for binary classification.  


## Training  

To address the severe class imbalance in the dataset, the data was **significantly downsampled**.  
Only **5 failures** were available in the provided dataset, and windows labeled as `0` (no failure) greatly outnumbered those labeled as `1` (failure).  

- **Data Split**  
  - 4 failure cases were used for **training and validation**.  
  - 1 failure case was held out for **testing**.  

- **Evaluation Strategy**  
  - After each training epoch, the model was evaluated on the test case.  
  - The performance on this test case was used as a criterion to judge whether the model was performing well or poorly.  
  
- **Tracked Metrics**  
  - `train_loss` and `val_loss` to monitor convergence.  
  - `train_acc`, `val_acc` and `f1` to measure classification accuracy.  

- **Training Management**  
  - **Early stopping** was applied to prevent overfitting and stop training when validation performance stopped improving.  
  - **Model checkpoints** were saved after each epoch

## Run scripts

---

### Option 1: Docker Setup

#### Prerequisites

- NVIDIA Container Toolkit (for GPU support)

#### Steps

1. Build Docker Image

```bash
docker build -t wt_failure_pred .
```

2. Run Container

```bash
docker run --gpus all -v $(pwd)/workspace:/workspace wt_failure_pred
```

3. Training

```bash
python3 train.py
```

4. Inference

```bash
python3 infer.py
```

---

### Option 2: Conda Setup

#### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- NVIDIA GPU drivers

#### Steps

1. Create Conda Environment

Create a new conda environment with Python 3.12:

```bash
conda create -n wt_failure_pred python=3.12
```

2. Activate Environment

```bash
conda activate wt_failure_pred
```

3. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

4. Training

Run the training script:

```bash
python train.py
```

5. Inference

Run the inference script:

```bash
python infer.py
```

---
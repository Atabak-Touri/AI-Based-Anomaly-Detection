# AI-Based Anomaly Detection for 3D Printing

## 📋 Overview

This project implements various machine learning and deep learning techniques for detecting anomalies in 3D printing processes. Using sensor data from accelerometers and tension measurements, the system can classify different types of printing failures and distinguish between proper and improper printing operations.

## 🎯 Problem Statement

3D printing processes can fail in multiple ways, leading to material waste, time loss, and poor quality outputs. This project aims to automatically detect and classify printing anomalies using AI-based approaches, enabling early intervention and quality assurance.

## 📊 Dataset

The data was collected by Joanna Sendorek et al. for 3D printing anomalies de-
tection. The data was collected during 3D printing process where several types of
annomalies were provoked and includes the temperature of working elements of the
printer, instruction force and the acceleration of printing head. The 3D printer used to
get data was Monkeyfab Spire, which was manufactured by Monkeyfab.
The dataset consists of sensor readings from 3D printing operations, including:

- **Accelerometer Data**: 
  - `accel0X`, `accel0Y`, `accel0Z` (First accelerometer, 3 axes)
  - `accel1X`, `accel1Y`, `accel1Z` (Second accelerometer, 3 axes)
- **Tension Measurements**: Force sensor readings
- **Timestamps**: Time-series data for temporal analysis

### Anomaly Categories

The system classifies the following categories:
1. **Proper**: Normal printing operation
2. **Arm Failure**: Mechanical arm malfunction
3. **Bowden**: Bowden tube issues
4. **Plastic**: Material/filament problems
5. **Retraction 0.5**: Retraction setting anomalies
6. **Unstick**: Bed adhesion failures

## 🔬 Case Studies

### 1. 1D-CNN Classification
- **File**: `CaseStudy_1D-CNN.ipynb`
- **Description**: Implements a 1D Convolutional Neural Network for multi-class anomaly classification
- **Features**:
  - Sequential CNN architecture with multiple convolutional layers
  - MaxPooling and Dropout for regularization
  - Z-score normalization
  - Data interpolation for missing values

### 2. 1D-CNN with RMS
- **File**: `CaseStudy_1D-CNN with RMS.ipynb`
- **Description**: Enhanced CNN model using Root Mean Square (RMS) feature transformation
- **Features**:
  - Windowed RMS calculation for noise reduction
  - Improved signal processing
  - Enhanced feature extraction

### 3. Binary Classification
- **File**: `CaseStudy_Binary Classification.ipynb`
- **Description**: Logistic Regression for Pass/Fail classification
- **Features**:
  - Simplified binary classification (Proper vs. All Failures)
  - SMOTE for handling class imbalance
  - Confusion matrix and classification metrics
  - Log loss evaluation

### 4. Feature Ablation Study
- **File**: `CaseStudy_Feature Ablation.ipynb`
- **Description**: Systematic analysis of feature importance
- **Features**:
  - Sequential feature removal experiments
  - Performance comparison across feature sets
  - Identification of most critical sensors
  - Model performance with reduced feature sets

### 5. Hyperparameter Tuning
- **File**: `CaseStudy_Hyperparameter_Tuning.ipynb`
- **Description**: Automated hyperparameter optimization using Keras Tuner
- **Features**:
  - Grid search for optimal model configuration
  - Tunable parameters: filters, kernel size, dropout rate, learning rate
  - Performance tracking across configurations
  - Best model selection

### 6. Kalman Filter
- **File**: `CaseStudy_KalmanFilter.ipynb`
- **Description**: Advanced signal processing using Kalman filtering
- **Features**:
  - Noise reduction in sensor readings
  - State estimation for time-series data
  - Comparison of raw vs. filtered data
  - Improved data quality for classification

### 7. Principal Component Analysis (PCA)
- **File**: `CaseStudy_PCA.ipynb`
- **Description**: Dimensionality reduction and feature extraction
- **Features**:
  - Variance analysis across principal components
  - Reduced feature space
  - Visualization of data distribution
  - Computational efficiency improvements

## 🛠️ Technologies & Libraries

```python
- Python 3.x
- TensorFlow / Keras (Deep Learning)
- scikit-learn (Machine Learning)
- pandas (Data Manipulation)
- numpy (Numerical Computing)
- matplotlib (Visualization)
- seaborn (Statistical Visualization)
- imbalanced-learn (SMOTE)
- keras-tuner (Hyperparameter Optimization)
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn imbalanced-learn keras-tuner
```

### Data Preparation

1. Place your dataset in the following structure:
```
dataset/
├── arm_failure/
│   ├── t.txt
│   └── j.json
├── bowden/
│   ├── t.txt
│   └── j.json
├── plastic/
│   ├── t.txt
│   └── j.json
├── proper/
│   ├── t.txt
│   └── j.json
├── retraction_05/
│   ├── t.txt
│   └── j.json
└── unstick/
    ├── t.txt
    └── j.json
```

2. Update the `base_directory` path in each notebook

### Running the Notebooks

1. Open Jupyter Notebook or JupyterLab
2. Navigate to the project directory
3. Open any case study notebook
4. Run cells sequentially

## 📈 Key Results & Insights

### Model Performance
- **1D-CNN**: Multi-class classification with high accuracy across all anomaly types
- **Binary Classification**: Effective Pass/Fail detection with balanced precision and recall
- **RMS Enhancement**: Improved noise resilience and feature stability

### Feature Importance
- Accelerometer data provides critical information for fault detection
- Tension measurements are particularly important for mechanical failures
- Multi-sensor fusion improves overall classification accuracy

### Preprocessing Impact
- Data interpolation effectively handles missing values
- Z-score normalization improves model convergence
- RMS windowing reduces noise while preserving signal characteristics
- Kalman filtering provides optimal signal smoothing

## 🔄 Workflow

```
Raw Sensor Data
    ↓
Data Loading & Preprocessing
    ↓
Feature Engineering (RMS, Kalman, PCA)
    ↓
Normalization (Z-score)
    ↓
Train/Test Split
    ↓
Model Training (CNN, Logistic Regression)
    ↓
Evaluation & Metrics
    ↓
Hyperparameter Tuning
    ↓
Final Model Deployment
```

## 📝 Data Processing Pipeline

1. **Data Loading**: Read CSV and JSON files for each category
2. **Time Alignment**: Filter data to start from printing initiation
3. **Interpolation**: Handle missing values using linear interpolation
4. **Feature Transformation**: Apply RMS, Kalman filtering, or PCA
5. **Normalization**: Z-score standardization
6. **Segmentation**: Create time-windowed sequences for CNN input
7. **Label Encoding**: Convert categorical labels to numerical format

## 🎓 Use Cases

- **Quality Control**: Real-time monitoring of 3D printing operations
- **Predictive Maintenance**: Early detection of mechanical failures
- **Process Optimization**: Identifying optimal printing parameters
- **Research**: Understanding failure modes in additive manufacturing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- Additional preprocessing techniques
- New classification models
- Enhanced visualization

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository.

## 📄 License

This project is available for educational and research purposes.

---

**Note**: Update the `base_directory` path in each notebook to match your local dataset location before running the experiments.

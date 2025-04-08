# Time Series Forecasting Comparison

## Overview
This repository contains a comprehensive comparison of various time series forecasting models, including traditional statistical methods and modern deep learning approaches. The project evaluates the performance of ARIMA, SARIMA, LSTM, GRU, and CNN models on synthetic time series data.

## Features
- Implementation of 5 different forecasting models:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - SARIMA (Seasonal ARIMA)
  - LSTM (Long Short-Term Memory networks)
  - GRU (Gated Recurrent Unit networks)
  - CNN (Convolutional Neural Networks)
- Performance evaluation using MAE and MSE metrics
- Visualization of prediction results
- Rolling forecast implementation for neural network models
- Synthetic time series data generation for testing

## Requirements
- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- TensorFlow 2.x
- scikit-learn
- statsmodels

## Installation
```bash
# Clone this repository
git clone https://github.com/yourusername/time-series-forecasting-comparison.git

# Navigate to the project directory
cd time-series-forecasting-comparison

# Install required packages
pip install numpy pandas matplotlib tensorflow scikit-learn statsmodels
```

## Usage
The main script can be run directly:
```bash
python time_series_comparison.py
```

This will:
1. Generate synthetic time series data (sine wave with noise)
2. Train all five forecasting models
3. Make predictions on the test set
4. Calculate performance metrics
5. Display visualizations comparing the models

## Model Details

### Statistical Models
- **ARIMA**: A classic time series forecasting method that combines autoregression, differencing, and moving average components
- **SARIMA**: An extension of ARIMA that incorporates seasonal patterns in the data

### Deep Learning Models
- **LSTM**: Recurrent neural networks designed to handle long-term dependencies in sequence data
- **GRU**: A simplified version of LSTM with fewer parameters but comparable performance
- **CNN**: Applies convolutional operations to capture local patterns in time series data

## Results
The repository includes visualization of:
- Bar charts comparing MAE and MSE for all models
- Line plots showing actual vs. predicted values for each model
- Detailed performance metrics summary

## Customization
You can customize the following parameters in the code:
- Time series data generation settings
- Train/test split ratio
- Window size for neural network models
- Neural network architecture (number of layers, units, etc.)
- ARIMA and SARIMA orders

## License
MIT

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.# Coparison_of_AI_and_other_techniques

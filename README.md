# Predicting Global Unemployment Rates using Artificial Neural Networks

## Overview

This project implements a deep learning model using Long Short-Term Memory (LSTM) neural networks to predict global unemployment rates. The model uses historical unemployment data from the World Bank to forecast future trends and analyze country-specific patterns.

## Features

- **LSTM-based Prediction Model**: Utilizes a Sequential neural network with LSTM layers to forecast global unemployment rates
- **World Bank Data Integration**: Automatically downloads and processes unemployment data from the World Bank API
- **Interactive Gradio Interface**: User-friendly web interface for exploring predictions and visualizations
- **Country-level Analysis**: Identifies countries with the highest unemployment rate increases
- **Visual Analytics**: Generates trend graphs and bar charts for data visualization

## Technology Stack

- **TensorFlow/Keras**: Deep learning framework for building and training the LSTM model
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Data preprocessing (MinMaxScaler)
- **Matplotlib**: Data visualization
- **Gradio**: Interactive web interface
- **Requests**: API data retrieval

## Dataset

The project uses the World Bank's unemployment indicator (SL.UEM.TOTL.ZS) which provides:
- Global unemployment rates by country
- Historical data spanning multiple years
- Country-level statistics for comparative analysis

## Model Architecture

- **Input Layer**: Sequences of 5 years of unemployment data
- **LSTM Layer**: 32 units for temporal pattern recognition
- **Dense Layer**: Single output neuron for unemployment rate prediction
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/saifullah2k3/Predicting-Global-Unemployment-Rates-using-Artificial-Neural-Networks.git
cd Predicting-Global-Unemployment-Rates-using-Artificial-Neural-Networks
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook "Predicting Global Unemployment Rates using Artificial Neural Networks.ipynb"
```

2. Run all cells to:
   - Download and process the World Bank unemployment data
   - Train the LSTM model
   - Launch the Gradio interface

3. Use the Gradio interface to:
   - Select two years to compare
   - View predicted global unemployment trends
   - Analyze countries with the highest rate increases
   - Explore interactive visualizations

## Project Structure

```
├── Predicting Global Unemployment Rates using Artificial Neural Networks.ipynb
├── requirements.txt
└── README.md
```

## Key Functionalities

### 1. Data Preprocessing
- Downloads unemployment data from World Bank API
- Cleans and reshapes data for time series analysis
- Computes global average unemployment rates per year

### 2. Model Training
- Prepares sequential data with a 5-year window
- Normalizes data using MinMaxScaler
- Trains LSTM model with 50 epochs

### 3. Predictions
- Forecasts future unemployment rates
- Generates year-by-year predictions
- Provides country-level comparative analysis

### 4. Visualization
- Global unemployment trend graphs
- Top 10 countries with highest increases
- Interactive charts and summaries

## Results

The model provides:
- Predicted global unemployment rates for future years
- Country-specific analysis and rankings
- Visual representations of trends and patterns
- Interactive exploration of historical and predicted data

## Limitations

- Predictions are based on historical patterns and may not account for unexpected economic events
- Model performance depends on data quality and availability
- Global averages may mask regional variations

## Future Improvements

- Implement more sophisticated LSTM architectures (e.g., bidirectional LSTM, stacked LSTM)
- Add more features (GDP, inflation, etc.) for multi-variate time series prediction
- Implement country-specific models for better regional accuracy
- Add confidence intervals for predictions
- Deploy the model as a web application

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- World Bank for providing the unemployment data
- TensorFlow team for the deep learning framework
- Gradio for the interactive interface framework

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. Predictions should not be used as the sole basis for economic or policy decisions.


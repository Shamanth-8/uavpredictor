# UAV IMU Predictor

A Streamlit web application for predicting UAV IMU data using ONNX models with CPU-only inference.

## Overview

This application provides a web interface for UAV IMU (Inertial Measurement Unit) prediction using a pre-trained ONNX model. The app loads a serialized model bundle and performs inference without requiring PyTorch at runtime.

## Features

- **CPU-only inference**: Uses ONNX Runtime for efficient CPU-based predictions
- **Web interface**: Built with Streamlit for easy interaction
- **Real-time predictions**: Instant IMU data predictions with demo inputs
- **Scalable architecture**: Uses cached model loading for optimal performance

## Requirements

- Python 3.8+
- Streamlit
- ONNX Runtime
- NumPy
- scikit-learn (for scalers)
- PyTorch (only for loading serialized model bundle)

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install streamlit onnxruntime numpy scikit-learn torch
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Files

- `app.py` - Main Streamlit application
- `uav_model.onnx` - ONNX model file for inference
- `uav_model_bundle.pkl` - Serialized model bundle containing scalers and metadata
- `README.md` - This documentation file

## Usage

1. Start the application with `streamlit run app.py`
2. The web interface will open in your browser
3. Click the "Predict" button to generate predictions using demo input data
4. View the predicted IMU values in the output section

## Model Details

The application uses:
- **ONNX Runtime**: For efficient model inference
- **Pre-trained model**: UAV IMU prediction model
- **Feature scaling**: Input/output normalization using scikit-learn scalers
- **Sequence processing**: Handles time-series IMU data sequences

## Architecture

The application follows a clean separation:
- **Model loading**: Cached resource loading for optimal performance
- **UI components**: Streamlit-based user interface
- **Inference pipeline**: ONNX Runtime for predictions
- **Post-processing**: Inverse scaling of model outputs

## Development

To extend the application:
- Add real IMU input methods (sliders, file upload, live data)
- Implement batch prediction capabilities
- Add visualization components for input/output data
- Include model performance metrics



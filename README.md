# Wrinkle-Prediction-via-Regression-Data-Driven-ML

## Overview
This project implements a data-driven machine learning approach to predict wrinkle patterns on a simulated 10x10 cloth grid using regression. It generates synthetic data with varying bend angles, frequencies, and phases, applies Principal Component Analysis (PCA) to reduce dimensionality, and trains a neural network to predict wrinkle shapes. The project includes both static visualization and real-time animation of predicted wrinkles.

## Files
- **data_gen.py**: Generates a synthetic dataset (`wrinkle_dataset.csv`) with 2000 samples, each containing input parameters (bend angle, frequency, phase) and 300 flattened vertex coordinates.
- **main.py**: Loads the dataset, applies PCA, trains a PyTorch neural network, evaluates the model, saves the trained model (`wrinkle_model.pth`) and PCA transformer (`pca_transformer.pkl`), and generates a static plot (`wrinkle_plot.png`) comparing actual vs. predicted wrinkles.
- **animate.py**: Loads the saved model and PCA to create a real-time animated 3D plot of the predicted wrinkle pattern by varying the phase parameter.
- **wrinkle_dataset.csv**: Synthetic dataset (regenerate with `data_gen.py` if needed; currently ~10.6 MB).
- **wrinkle_model.pth**: Saved neural network weights.
- **pca_transformer.pkl**: Saved PCA transformer for dimensionality reduction.
- **wrinkle_plot.png**: Static visualization of actual vs. predicted wrinkles.

## Setup and Usage
1. **Install Dependencies**: pip install numpy pandas scikit-learn torch matplotlib joblib
2. **Generate Data**: python data_gen.py
3. **Train Model and Visualize**: python main.py - This saves the model, PCA, and a static plot.
4. **Run Animation**: Displays a real-time animation of the wrinkle pattern.

## Results
- **Training**: 1000 epochs, final test loss typically around 0.05 (varies with random seed).
- **Static Visualization**: Side-by-side 3D plots of actual vs. predicted wrinkles (saved as `wrinkle_plot.png`).
- **Animation**: Real-time waving pattern based on phase variation.

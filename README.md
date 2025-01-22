# NEC Activity 1

This project focuses on implementing a Back Propagation algorithm and evaluating its predictive performance by comparing it against two alternative approaches:  
- Back Propagation implemented with PyTorch  
- Multi-Linear Regression  

## Dataset

The dataset used for this regression task aims to predict flight prices. It was originally sourced from [Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).  
To simplify the analysis and reduce computational requirements, a subset of 5,000 rows was randomly selected from the full dataset.

## Project Structure

The project is organized as follows:  
- **`data_preprocessing.ipynb`**: Preprocesses the raw dataset to clean and prepare it for modeling.  
- **`multi_linear_regression.ipynb`**: Implements flight price prediction using Sci-kit Learn's multi-linear regression.  
- **`neural_network_pytorch.ipynb`**: Predicts flight prices using a neural network built with PyTorch.  
- **`neural_network_pytorch_regularization.ipynb`**: Predicts flight prices using a neural network built with PyTorch, applying regularization techniques.
- **`neural_network_self.ipynb`**: Flight price prediction using a custom-built neural network from scratch.  
- **`neural_network_self_crossval.ipynb`**: Flight price prediction using a custom-built neural network from scratch (with k-fold cross-validation).
- **`NeuralNet.py`**: Contains the implementation of the custom neural network.  
- **`NeuralNetCross.py`**: Contains the implementation of the custom neural network supporting cross-validation.
- **`utils.py`**: Provides utility functions for plotting and error calculation.  

## Directories

- **`models/`**: Stores trained models saved in `.pkl` format for our self-implemented neural network.  
- **`plots/`**: Contains visualizations including loss curves and scatter plots showing the comparison between true and predicted values for various hyperparameter configurations.  
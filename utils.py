from typing import Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


def print_errors(mse: float, mae: float, mape: float) -> None:
    """
    Print common regression error metrics.

    Displays the following metrics to evaluate the performance of a regression model:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Mean Absolute Percentage Error (MAPE)

    Results are rounded to two decimal places for improved readability.

    :param mse: float
        The mean squared error between actual and predicted values.
    :param mae: float
        The mean absolute error between actual and predicted values.
    :param mape: float
        The mean absolute percentage error between actual and predicted values.

    :return: None
    """
    print('Mean Squared Error (MSE):', mse)
    print('Mean Absolute Error (MAE):', mae)
    print('Mean Absolute Percentage Error (MAPE):', mape)


def print_std(mse: float, mae: float, mape: float) -> None:
    """
    Print the standard deviations of common regression error metrics.

    Displays the following metrics to assess variability in model performance:
    - Standard deviation of Mean Squared Error (MSE)
    - Standard deviation of Mean Absolute Error (MAE)
    - Standard deviation of Mean Absolute Percentage Error (MAPE)

    :param mse: float
        Standard deviation of the mean squared error.
    :param mae: float
        Standard deviation of the mean absolute error.
    :param mape: float
        Standard deviation of the mean absolute percentage error.

    :return: None
    """
    print('Standard Deviation (MSE):', mse)
    print('Standard Deviation (MAE):', mae)
    print('Standard Deviation (MAPE):', mape)


def compute_errors(y_true: Any, y_pred: Any) -> tuple[float, float, float]:
    """
    Compute common regression error metrics.

    Calculates the following metrics for evaluating the performance of a regression model:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Mean Absolute Percentage Error (MAPE)

    Results are rounded to two decimal places.

    :param y_true: array-like
        True target values.
    :param y_pred: array-like
        Predicted target values.

    :return: tuple[float, float, float]
        A tuple containing MSE, MAE, and MAPE in that order.
    """
    mse = round(mean_squared_error(y_true, y_pred), 2)
    mae = round(mean_absolute_error(y_true, y_pred), 2)
    mape = round(mean_absolute_percentage_error(y_true, y_pred), 2)

    return mse, mae, mape


def scatter(y_true: Any, y_pred: Any) -> None:
    """
    Create a scatter plot comparing actual and predicted values.

    Visualizes the relationship between the true target values (`y_true`) and
    the predicted values (`y_pred`) using a scatter plot. Includes a reference
    line representing ideal predictions for better performance assessment.

    Plot Features:
    - Blue points represent predicted vs actual values.
    - A red dashed line indicates the ideal prediction line (y = x).
    - Labeled axes, title, and a grid for better clarity.

    :param y_true: array-like
        True target values.
    :param y_pred: array-like
        Predicted target values.

    :return: None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal Prediction')
    plt.xlabel('Actual Scaled Prices (y_test)')
    plt.ylabel('Predicted Scaled Prices (y_predicted)')
    plt.title('Comparison of Predicted vs Actual Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

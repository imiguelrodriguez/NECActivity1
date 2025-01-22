from typing import Tuple, Any

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


def print_errors(mse, mae, mape) -> None:
    """
        Print common regression error metrics.

        Calculates and displays the following metrics to evaluate the performance
        of a regression model:
        - Mean Squared Error (MSE)
        - Mean Absolute Error (MAE)
        - Mean Absolute Percentage Error (MAPE)

        Results are rounded to two decimal places for improved readability.

        :param y_true: array-like
            True target values.
        :param y_pred: array-like
            Predicted target values.

        :return: None
    """

    print('mean_squared_error : ', mse)
    print('mean_absolute_error : ', mae)
    print('mean_absolute_percentage_error : ', mape)

def compute_errors(y_true, y_pred) -> tuple[Any, Any, Any]:
    mse = round(mean_squared_error(y_true, y_pred), 2)
    mae = round(mean_absolute_error(y_true, y_pred), 2)
    mape = round(mean_absolute_percentage_error(y_true, y_pred), 2)

    return mse, mae, mape

def scatter(y_true, y_pred) -> None:
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

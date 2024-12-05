from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def print_errors(y_true, y_pred):
  print('mean_squared_error : ', round(mean_squared_error(y_true, y_pred),2))
  print('mean_absolute_error : ', round(mean_absolute_error(y_true, y_pred),2))
  print('mean_absolute_percentage_error : ', round(mean_absolute_percentage_error(y_true, y_pred),2))

def scatter(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal Prediction')
    plt.xlabel('Actual Scaled Prices (y_test)')
    plt.ylabel('Predicted Scaled Prices (y_predicted)')
    plt.title('Comparison of Predicted vs Actual Prices')
    plt.legend()
    plt.grid(True)
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def csv_reader(path, flag):
    if flag == 'x':
        str_x = path + "features.csv"
        data = pd.read_csv(str_x).values.astype(np.float64)
    elif flag == 'y':
        str_y = path + "targets.csv"
        data = pd.read_csv(str_y).values.astype(np.float64)
    else:
        str_fold = path + "folds.csv"
        data = pd.read_csv(str_fold).values.astype(np.float64)
    return data


abs_x = csv_reader("abs_", 'x')
abs_y = csv_reader("abs_", 'y')
abs_folds = csv_reader("abs_", 'folds')

linear_x = csv_reader("linear_", 'x')
linear_y = csv_reader("linear_", 'y')
linear_folds = csv_reader("linear_", 'folds')

sin_x = csv_reader("sin_", 'x')
sin_y = csv_reader("sin_", 'y')
sin_folds = csv_reader("sin_", 'folds')

x1 = abs_x[:, 0]
upper_y1 = abs_y[:, 0]
lower_y1 = abs_y[:, 1]

x2 = sin_x[:, 0]
upper_y2 = sin_y[:, 0]
lower_y2 = sin_y[:, 1]

x3 = linear_x[:,0]
upper_y3 = linear_y[:, 0]
lower_y3 = linear_y[:, 1]

x1_sort = np.argsort(x1)
x2_sort = np.argsort(x2)
x3_sort = np.argsort(x3)

x1_sorted = x1[x1_sort]
x2_sorted = x2[x2_sort]
x3_sorted = x3[x3_sort]

fun_abs = np.abs(x1_sorted-5)
fun_sin = np.sin(x2_sorted)
fun_linear = x3_sorted/5

plt.figure(figsize=(7, 2))
plt.subplot(131)
plt.title("f(x) = |x|")
plt.scatter(x1, upper_y1, edgecolor='grey', facecolor='grey', linewidth=1.0, s=10)
plt.scatter(x1, lower_y1, edgecolor='grey', facecolor='none', linewidth=1.0, s=10)
plt.plot(x1_sorted, fun_abs, color="black", linewidth=2, linestyle='-',zorder=1)

plt.subplot(132)
plt.title("f(x) = sin(x)")
plt.scatter(x2, upper_y2, edgecolor='grey', facecolor='grey', linewidth=1.0, s=10)
plt.scatter(x2, lower_y2, edgecolor='grey', facecolor='none', linewidth=1.0, s=10)
plt.plot(x2_sorted, fun_sin, color="black", linewidth=2, linestyle='-', zorder=1)

plt.subplot(133)
plt.title("f(x) = x/5")
plt.scatter(x3, upper_y3, edgecolor='grey', facecolor='grey', linewidth=1.0, s=10)
plt.scatter(x3, lower_y3, edgecolor='grey', facecolor='none', linewidth=1.0, s=10)
plt.plot(x3_sorted, fun_linear, color="black", linewidth=2, linestyle='-', zorder=1)
plt.show()
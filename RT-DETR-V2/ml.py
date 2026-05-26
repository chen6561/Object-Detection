import numpy as np
import matplotlib.pyplot as plt

# 西瓜数据集3.0（密度，含糖率）
data = np.array([
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318],
    [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211],
    [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
    [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042],
    [0.719, 0.103]
])
X = data[:, 0:1]  # 密度作为特征
y = data[:, 1]    # 含糖率作为标签

# 单变量线性回归
def linear_regression_single(X, y):
    m = len(X)
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    # 计算w
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)
    w = numerator / denominator
    # 计算b
    b = y_mean - w * x_mean
    return w, b

w, b = linear_regression_single(X, y)
print(f"单变量线性回归结果：w={w:.4f}, b={b:.4f}")

# 绘制拟合直线
plt.scatter(X, y, c='red', marker='o', label='样本点')
x_line = np.linspace(0.2, 0.8, 100)
y_line = w * x_line + b
plt.plot(x_line, y_line, c='blue', label='拟合直线')
plt.xlabel('密度')
plt.ylabel('含糖率')
plt.legend()
plt.show()

# 多元线性回归（加入一个虚拟特征演示）
X_multi = np.hstack((X, np.random.randn(len(X), 1)))  # 加入一个随机特征
def linear_regression_multi(X, y):
    m = len(X)
    # 加入偏置项
    X_b = np.hstack((X, np.ones((m, 1))))
    # 计算闭式解
    w_hat = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return w_hat

w_hat = linear_regression_multi(X_multi, y)
print(f"多元线性回归结果：w={w_hat[:-1]}, b={w_hat[-1]:.4f}")
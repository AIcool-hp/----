import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 加载Iris数据集
iris = load_iris()
X, y = iris.data, iris.target

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 输出PCA结果的形状
print("PCA降维后的数据形状:", X_pca.shape)

# 输出每个主成分的方差比例
explained_variance = pca.explained_variance_ratio_
print("各主成分的方差比例：", explained_variance)
print("累计方差比例：", np.cumsum(explained_variance))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 可视化PCA结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], 
                      X_pca[:, 1], 
                      c=y, 
                      cmap='viridis', 
                      edgecolor='k', 
                      s=100)
plt.title('Iris数据集的PCA降维结果')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.colorbar(scatter, label='类别标签', ticks=[0, 1, 2])
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

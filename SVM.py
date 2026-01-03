from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建与训练SVM分类器
svm_model = svm.SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    random_state=42
)

svm_model.fit(X_train, y_train)

# 预测与评估
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("===== SVM 模型评估 =====")
print(f"测试集准确率：{accuracy:.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 可视化（混淆矩阵）
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True,
            fmt='d', 
            cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('SVM分类器混淆矩阵')
plt.show()

# 可视化不同核函数对准确率的影响
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
accuracies = []
for kernel in kernels:
    svm_k = svm.SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    svm_k.fit(X_train, y_train)
    y_k_pred = svm_k.predict(X_test)
    acc = accuracy_score(y_test, y_k_pred)
    accuracies.append(acc)

plt.figure(figsize=(10, 6))
plt.bar(kernels, accuracies, color='skyblue')
plt.xlabel('核函数类型')
plt.ylabel('准确率')
plt.title('不同核函数对SVM分类器准确率的影响')
plt.show()

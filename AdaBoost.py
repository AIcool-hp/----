from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
data = load_iris()
X, y = data.data, data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建AdaBoost分类器
adaboost = AdaBoostClassifier(
    n_estimators=20,  
    learning_rate=1.0,  
    random_state=42
)

# 训练模型
adaboost.fit(X_train, y_train)

# 预测和评估
y_pred = adaboost.predict(X_test)
print("准确率:", accuracy_score(y_test, y_pred))
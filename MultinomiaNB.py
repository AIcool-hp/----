from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
import numpy as np
import matplotlib.pyplot as plt


# 数据集路径
train_dir = r"C:\Users\15614\Desktop\20news-bydate\20news-bydate-train"
test_dir = r"C:\Users\15614\Desktop\20news-bydate\20news-bydate-test"

# 加载训练集和测试集
train_data = load_files(train_dir, encoding='latin1', decode_error='ignore')
test_data = load_files(test_dir, encoding='latin1', decode_error='ignore')

# 构建朴素贝叶斯分类器pipelines
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
model = MultinomialNB()
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', model)
])

# 训练分类器
pipeline.fit(train_data.data, train_data.target)

# 评估分类器
accuracy = pipeline.score(test_data.data, test_data.target)
print("分类器准确率:", accuracy)

# 预测示例
sample_text = ["I love programming and machine learning"]
prediction = pipeline.predict(sample_text)
print("示例预测结果:", train_data.target_names[prediction[0]])

# 输出每个类别的后验概率
posterior_probs = pipeline.predict_proba(sample_text)
print("每个类别的后验概率:", posterior_probs)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 可视化后验概率
plt.figure(figsize=(12, 6))  
plt.bar(train_data.target_names, 
        posterior_probs[0], 
        color='skyblue', 
        label='后验概率')
plt.title('每个类别的后验概率')
plt.xlabel('类别')
plt.ylabel('后验概率')
plt.xticks(rotation=45, fontsize=8)  
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)  
plt.tight_layout() 
plt.show()

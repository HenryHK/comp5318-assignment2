import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

sample = np.loadtxt('data/train-sample-0.csv', delimiter=',')
target = np.loadtxt('data/train-target-0.csv', delimiter=',')
predict_sample = np.loadtxt('data/predict-sample-0.csv', delimiter=',')
predict_target = np.loadtxt('data/predict-target-0.csv', delimiter=',')

classifier = LogisticRegression()
classifier.fit(sample, target)  # 训练数据来学习，不需要返回值

result = classifier.predict(predict_sample)

print(metrics.classification_report(result, predict_target))

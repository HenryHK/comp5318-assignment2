import numpy as np
from sklearn import ensemble
from sklearn import metrics

sample = np.loadtxt('data/train-sample-0.csv', delimiter=',')
target = np.loadtxt('data/train-target-0.csv', delimiter=',')
predict_sample = np.loadtxt('data/predict-sample-0.csv', delimiter=',')
predict_target = np.loadtxt('data/predict-target-0.csv', delimiter=',')

rf = ensemble.RandomForestClassifier()
rf.fit(sample, target)  # 训练数据来学习，不需要返回值

result = classifier.predict(predict_sample)
result_metrics = metrics.classification_report(result, predict_target)

with open('result/rf.txt', 'w') as output:
    output.write(result_metrics)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

sample = np.loadtxt('data/train-sample-0.csv', delimiter=',')
target = np.loadtxt('data/train-target-0.csv', delimiter=',')
predict_sample = np.loadtxt('data/predict-sample-0.csv', delimiter=',')
predict_target = np.loadtxt('data/predict-target-0.csv', delimiter=',')

knn = KNeighborsClassifier()
knn.fit(sample, target)

result = knn.predict(predict_sample)
result_metrics = metrics.classification_report(result, predict_target)

with open('result/knn.txt', 'w') as output:
    output.write(result_metrics)

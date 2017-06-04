import numpy as np
from sklearn import ensemble
from sklearn import metrics

sample = np.loadtxt('data/train-sample-2.csv', delimiter=',')
target = np.loadtxt('data/train-target-2.csv', delimiter=',')
predict_sample = np.loadtxt('data/predict-sample-2.csv', delimiter=',')
predict_target = np.loadtxt('data/predict-target-2.csv', delimiter=',')

rf = ensemble.RandomForestClassifier()
rf.fit(sample, target)

result = rf.predict(predict_sample)
result_metrics = metrics.classification_report(result, predict_target)

with open('result/rf.txt', 'w') as output:
    output.write(result_metrics)

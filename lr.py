import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

sample = np.loadtxt('data/train-sample-2.csv', delimiter=',')
target = np.loadtxt('data/train-target-2.csv', delimiter=',')
predict_sample = np.loadtxt('data/predict-sample-2.csv', delimiter=',')
predict_target = np.loadtxt('data/predict-target-2.csv', delimiter=',')

classifier = LogisticRegression()
classifier.fit(sample, target)

print("sample size: "+str(len(sample)))

result = classifier.predict(predict_sample)
metrics_result = metrics.classification_report(result, predict_target)

with open('lr.txt', 'w') as output:
    output.write(metrics_result)
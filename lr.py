import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time

for i in range(10):
    sample = np.loadtxt('data/train-sample-'+str(i)+'.csv', delimiter=',')
    target = np.loadtxt('data/train-target-'+str(i)+'.csv', delimiter=',')
    predict_sample = np.loadtxt('data/predict-sample-'+str(i)+'.csv', delimiter=',')
    predict_target = np.loadtxt('data/predict-target-'+str(i)+'.csv', delimiter=',')

    begin = time.time()

    classifier = LogisticRegression()
    classifier.fit(sample, target)

    print("sample size: "+str(len(sample)))

    result = classifier.predict(predict_sample)

    end = time.time()

    metrics_result = metrics.classification_report(result, predict_target)

    with open('result/lr.txt', 'a') as output:
        output.write(metrics_result)
        output.write("\n")
        output.write("Use: "+str(end-begin)+"s")
        output.write('------------------------------------\n')
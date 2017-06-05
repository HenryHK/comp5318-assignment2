import numpy as np
from sklearn import ensemble
from sklearn import metrics
import time


for i in range(10):
    sample = np.loadtxt('data/train-sample-'+str(i)+'.csv', delimiter=',')
    target = np.loadtxt('data/train-target-'+str(i)+'.csv', delimiter=',')
    predict_sample = np.loadtxt('data/predict-sample-'+str(i)+'.csv', delimiter=',')
    predict_target = np.loadtxt('data/predict-target-'+str(i)+'.csv', delimiter=',')

    begin = time.time()

    rf = ensemble.RandomForestClassifier()
    rf.fit(sample, target)

    result = rf.predict(predict_sample)

    end = time.time()

    print(end-begin)

    result_metrics = metrics.classification_report(result, predict_target)

    with open('result/rf.txt', 'a') as output:
        output.write(result_metrics)
        output.write("\n")
        output.write("Use: "+str(end-begin)+"s")
        output.write('------------------------------------\n')

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time

sample = np.loadtxt('data/train-sample-2.csv', delimiter=',')
target = np.loadtxt('data/train-target-2.csv', delimiter=',')
predict_sample = np.loadtxt('data/predict-sample-2.csv', delimiter=',')
predict_target = np.loadtxt('data/predict-target-2.csv', delimiter=',')

begin = time.time()

knn = KNeighborsClassifier()
knn.fit(sample, target)

result = knn.predict(predict_sample)

end = time.time()


result_metrics = metrics.classification_report(result, predict_target)

with open('result/knn.txt', 'w') as output:
    output.write(result_metrics)
    output.write("\n")
    output.write("Use: "+str(end-begin)+"s")

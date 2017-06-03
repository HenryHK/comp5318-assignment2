import numpy as np
from sklearn.cluster import KMeans
from operator import itemgetter, attrgetter
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def e_distance(x, y):
    """
    @param x: point.
    @param y: point.
    
    @return euclidean distance between this point.
    """
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x-y)


sample = np.loadtxt('data/train-sample-0.csv', delimiter=',')
target = np.loadtxt('data/train-target-0.csv', delimiter=',')
predict_sample = np.loadtxt('data/predict-sample-0.csv', delimiter=',')
predict_target = np.loadtxt('data/predict-target-0.csv', delimiter=',')

k = 7
kmeans = KMeans(n_clusters=k).fit(sample)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print(labels)
print(centroids)

sample_with_distance=[]
target_with_distance=[]

i=0
for each in sample:
    label = labels[i]
    center = centroids[label]
    distance = e_distance(each, center)
    sample_with_distance.append((each,distance))
    target_with_distance.append((target[i],distance))
    i += 1

sample_with_distance.sort(key=itemgetter(1),reverse=True)
target_with_distance.sort(key=itemgetter(1),reverse=True)

length = len(sample_with_distance)
sample_with_distance = sample_with_distance[0:int(length*0.9)]
target_with_distance = target_with_distance[0:int(length*0.9)]

sample = []
target = []

for data, distance in sample_with_distance:
    sample.append(data)

for data, distance in target_with_distance:
    target.append(data)

classifier = LogisticRegression()
classifier.fit(sample, target)

result = classifier.predict(predict_sample)

result_metrics = metrics.classification_report(result, predict_target)

with open('result/lr_with_orc.txt', 'w') as output:
    output.write(result_metrics)

# kmeans.fit(sample)

# with open('result/knn.txt', 'w') as output:
#     output.write(result_metrics)

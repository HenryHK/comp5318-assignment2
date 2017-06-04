import numpy as np
from sklearn.cluster import KMeans
from operator import itemgetter, attrgetter
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import collections

def e_distance(x, y):
    """
    @param x: point.
    @param y: point.
    
    @return euclidean distance between this point.
    """
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x-y)


sample = np.loadtxt('data/train-sample-2.csv', delimiter=',')
target = np.loadtxt('data/train-target-2.csv', delimiter=',')
predict_sample = np.loadtxt('data/predict-sample-2.csv', delimiter=',')
predict_target = np.loadtxt('data/predict-target-2.csv', delimiter=',')

original_length = len(sample)
print("sample size: "+str(original_length))

k = 7
kmeans = KMeans(n_clusters=k).fit(sample)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

combined_data = zip(sample, target)

sample_with_distance=[]
target_with_distance=[]

d = collections.defaultdict(list)
outlier = 0.95

i=0
for each in combined_data:
    label = labels[i]
    center = centroids[label]
    distance = e_distance(each[0], center)
    d[label].append((each, distance))
    i += 1

sample = []
target = []

for key, values in d.items():
    length = len(values)
    values.sort(key= itemgetter(1), reverse = False)
    values = values[0:int(length*outlier)]
    for each in values:
        sample.append(each[0][0])
        target.append(each[0][1])

# sample_with_distance.sort(key=itemgetter(1),reverse=False)
# target_with_distance.sort(key=itemgetter(1),reverse=False)

# outlier = 0.95

# length = len(sample_with_distance)
# sample_with_distance = sample_with_distance[0:int(length*outlier)]
# target_with_distance = target_with_distance[0:int(length*outlier)]

# sample = []
# target = []

# for data, distance in sample_with_distance:
#     sample.append(data)

# for data, distance in target_with_distance:
#     target.append(data)

# new_length = len(sample)

# print("Remove "+str(original_length-new_length)+" outliers")

# lr begins

classifier = LogisticRegression()
classifier.fit(sample, target)

result = classifier.predict(predict_sample)

result_metrics = metrics.classification_report(result, predict_target)

with open('result/lr_with_orc.txt', 'w') as output:
    output.write(result_metrics)

# kmeans.fit(sample)

# with open('result/knn.txt', 'w') as output:
#     output.write(result_metrics)
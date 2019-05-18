import csv
import sys
import math
import random
import matplotlib.pyplot as plt
import numpy

class Cluster(object):
    def __init__(self, centroid):
        self.centroid = centroid
        self.members = []

    def add_member(self, member):
        self.members.append(member)

    def change_centroid(self, centroid):
        self.centroid = centroid

    def get_wcscore(self, manhattan):
        score = 0
        for instance in self.members:
            distance = get_distance(self.centroid, instance, manhattan)
            distance = math.pow(distance, 2)
            score = score + distance
        return score

def load_csv(fname):
    lines = csv.reader(open(fname, "r"))
    next(lines, None)
    data = list(lines)
    entry_length = len(data[0])
    for i in range(len(data)):
        attr_index = 0
        while attr_index < entry_length:
            if attr_index < 3:
                data[i].pop(0)
            elif attr_index == 5:
                data[i].pop(2)
            elif attr_index > 7:
                data[i].pop(4)
            attr_index += 1
        for index, value in enumerate(data[i]):
            data[i][index] = float(value)
    return data

def get_distance(first, second, manhattan):
    dist = 0
    if manhattan:
        for index, value in enumerate(first):
            dist += value - second[index]
    else:
        dist = 0
        for index, value in enumerate(first):
            dist += math.pow(value - second[index], 2)
        dist = math.sqrt(dist)
    return dist

def random_centroids(data, k):
    points = random.sample(data, k)
    clusters = []
    for centroid in points:
        cluster = Cluster(centroid)
        clusters.append(cluster)
    return clusters

def connect_with_clusters(data, clusters, manhattan):
    for instance in data:
        min_dist = 9999999.99
        centroid_index = 0
        for index, cluster in enumerate(clusters):
            dist = get_distance(cluster.centroid, instance, manhattan)
            if min_dist > dist:
                min_dist = dist
                centroid_index = index
        clusters[centroid_index].add_member(instance)

def change_clusters_with_data(clusters):
    for index, cluster in enumerate(clusters):
        mean_centroid = [0, 0, 0, 0]
        for entry in cluster.members:
            for i, val in enumerate(entry):
                mean_centroid[i] += val/len(cluster.members)
        clusters[index].change_centroid(mean_centroid)

def change_members_new_clusters(clusters, manhattan):
    change_count = 0
    for i, main_cluster in enumerate(clusters):
        for instance in main_cluster.members:
            min_dist = get_distance(main_cluster.centroid, instance, manhattan)
            result_index = i
            for index, sub_cluster in enumerate(clusters):
                dist = get_distance(sub_cluster.centroid, instance, manhattan)
                if min_dist > dist:
                    min_dist = dist
                    result_index = index
            if result_index != i:
                main_cluster.members.remove(instance)
                clusters[result_index].members.append(instance)
                change_count += 1
    return change_count

def log_load_csv(filename): # we eliminate all unnecessary attributes
    lines = csv.reader(open(filename, "rb"))
    next(lines, None)
    data = list(lines)
    entry_length = len(data[0])
    for i in range(len(data)):
        attr_index = 0
        while attr_index < entry_length:
            if attr_index < 3:
                data[i].pop(0)
            elif attr_index == 5:
                data[i].pop(2)
            elif attr_index > 7:
                data[i].pop(4)
            attr_index += 1
        for idx, value in enumerate(data[i]):
            if idx == 2 | idx == 3:
                data[i][idx] = math.log(float(value))
            else:
                data[i][idx] = float(value)
    return data

def std_load_csv(filename):
    lines = csv.reader(open(filename, "rb"))
    next(lines, None)
    data = list(lines)
    entry_length = len(data[0])
    for i in range(len(data)):
        attr_index = 0
        while attr_index < entry_length:
            if attr_index < 3:
                data[i].pop(0)
            elif attr_index == 5:
                data[i].pop(2)
            elif attr_index > 7:
                data[i].pop(4)
            attr_index += 1
        for idx, value in enumerate(data[i]):
            data[i][idx] = float(value)
    std_set = []
    mean_set = []
    total_array = {0: [], 1: [], 2: [], 3: []}
    for instance in data:
        for i, value in enumerate(instance):
            total_array[i].append(value)
    for i, attr in enumerate(total_array):
        std_set.append(numpy.std(total_array[i]))
        mean_set.append(numpy.mean(total_array[i]))
    for i, instance in enumerate(data):
        for idx, attr_val in enumerate(data[i]):
            data[i][idx] = abs((data[i][idx] - mean_set[idx])/std_set[idx])
    return data

def plot_latitude(clusters, is_review):
    for cl in clusters:
        color = numpy.random.rand(3, 1)
        if is_review:
            plt.scatter(cl.centroid[2], cl.centroid[3], marker="x", c=color, s=100)
        else:
            plt.scatter(cl.centroid[0], cl.centroid[1], marker="x", c=color, s=100)
        x_vals = []
        y_vals = []
        for entry in cl.members:
            if is_review:
                x_vals.append(entry[2])
                y_vals.append(entry[3])
            else:
                x_vals.append(entry[0])
                y_vals.append(entry[1])
        plt.scatter(x_vals, y_vals, marker="o", c=color, s=25)
    if is_review:
        plt.xlabel("Review Count")
        plt.ylabel("Checkins")
    else:
        plt.xlabel("Latitude")
        plt.ylabel("Longtitude")
    plt.show()

def main():
    k = int(sys.argv[2])
    args = int(sys.argv[3])

    if args == 2:
        data = log_load_csv(sys.argv[1])
    elif args == 3:
        data = std_load_csv(sys.argv[1])
    else:
        data = load_csv(sys.argv[1])

    clusters = random_centroids(data, k)

    if args == 1:
        manhattan = False
        connect_with_clusters(data, clusters, manhattan)
        change_clusters_with_data(clusters)

        while change_members_new_clusters(clusters, manhattan) != 0:
            change_clusters_with_data(clusters)

        score = 0
        for i, cluster in enumerate(clusters):
            score += cluster.get_wcscore(manhattan)
        print("WC-SSE=" + str(score))

        for i, cluster in enumerate(clusters):
            print("Centroid" + str(i + 1) + "=" + str(cluster.centroid))

    elif args == 4:
        manhattan = True
        connect_with_clusters(data, clusters, manhattan)
        change_clusters_with_data(clusters)

        while change_members_new_clusters(clusters, manhattan) != 0:
            change_clusters_with_data(clusters)

        score = 0
        for i, cluster in enumerate(clusters):
            score += cluster.get_wcscore(manhattan)
        print("WC-SSE=" + score)

        for i, cluster in enumerate(clusters):
            print("Centroid" + str(i + 1) + "=" + str(cluster.centroid))

    elif args == 5:
        manhattan = False
        percentage = int(len(data) * 0.01)
        sample = random.sample(data, percentage)

        clusters = random_centroids(sample, k)
        connect_with_clusters(sample, clusters, manhattan)
        change_clusters_with_data(clusters)

        while change_members_new_clusters(clusters, manhattan) != 0:
            change_clusters_with_data(clusters)

        score = 0
        for i, cluster in enumerate(clusters):
            score += cluster.get_wcscore(manhattan)
        print("WC-SSE=" + str(score))

        for i, cluster in enumerate(clusters):
            print("Centroid" + str(i + 1) + "=" + str(cluster.centroid))

    #plot_latitude(clusters, is_review=True)
    plt.plot(str(clusters))

main()
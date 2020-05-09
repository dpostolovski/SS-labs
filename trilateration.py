import math
import random

import numpy
from matplotlib import pyplot
from scipy.optimize import fmin_powell, least_squares

class Node():
    def __init__(self, true_position, calculated_position, level=None, heuristic =None):
        self.level = level
        self.calculated_position = calculated_position
        self.true_position = true_position
        self.heuristic = heuristic

class Simulator():
    def __init__(self,nodes, anchor_nodes, length, radio_range, noise ):
        self.noise = noise
        self.radio_range = radio_range
        self.length = length
        self.anchor_nodes = anchor_nodes
        self.n = nodes

        self.nodes = list()

        self.initialize_nodes(nodes, anchor_nodes, length)


    def initialize_nodes(self, n, anchor_nodes, length):
        for i in range(anchor_nodes):
            coordinates = self.random_coordinates(0, length)
            node = Node(coordinates, coordinates, 0)
            self.nodes.append(node)

        for i in range(n - anchor_nodes):
            coordinates = self.random_coordinates(0, length)
            node = Node(coordinates, (None, None), None)
            self.nodes.append(node)

    def random_coordinates(self, start, end):
        x = random.uniform(start,end)
        y = random.uniform(start,end)
        return x, y

    def run(self, iteration=0):
        for level in range(iteration):
            anchors = [item for item in sim.nodes if item.level is not None and item.level <= level]
            for node in self.nodes:
                if node.level is None:
                    anchors_for_node = self.find_anchors_of(node, anchors)
                    if anchors_for_node.__len__() < 3:
                        continue

                    position = self._localize_node(node, anchors_for_node)
                    node.calculated_position = position[0], position[1]
                    node.level = level+1
                    node.heuristic = self.heuristic(node, anchors_for_node)

    def heuristic(self, node, anchors_for_node):
        raise NotImplemented

    def find_anchors_of(self, node, anchors):
        raise NotImplemented

    @staticmethod
    def calculate_distance(a, b):
        x1,y1 = a
        x2,y2 = b
        return math.sqrt( math.pow(x1-x2 ,2) + math.pow(y1 - y2, 2) )

    @staticmethod
    def distfunc(L, points):
        distsum = 0
        for c in points:
            dist = Simulator.calculate_distance((L[0], L[1]), (c[0], c[1])) - c[2]
            distsum += dist * dist
        return distsum

    def _localize_node(self,node, anchors):
        points  = list()
        for anchor in anchors:
            x, y = anchor.true_position
            distance = Simulator.calculate_distance(node.true_position, anchor.calculated_position)
            distance = distance + random.uniform(-distance*self.noise, distance*self.noise )
            points.append([x, y, distance])

        anchor =points[0]

        L = fmin_powell(Simulator.distfunc, numpy.array([anchor[0], anchor[1], 0]), args=(points,), xtol=0.01, ftol=0.01, disp=False)
        return L

    def metrics(self):
        localized_nodes = len([ node for node in self.nodes if node.level is not None ]) - self.anchor_nodes
        non_anchor_nodes = self.n - self.anchor_nodes
        localized_ratio = localized_nodes / non_anchor_nodes

        errors = [ self.calculate_distance(node.calculated_position, node.true_position)
                   for node in self.nodes if node.level is not None and node.level !=0 ]

        ale = sum(errors) / ((self.n - self.anchor_nodes) * self.radio_range )
        return localized_ratio, ale

class ClosestAnchorsSimulator(Simulator):
    def find_anchors_of(self, node, anchors):
        distances = list()
        for anchor in anchors:
            distance = self.calculate_distance(node.true_position, anchor.true_position)
            if distance > self.radio_range:
                continue
            distances.append((distance, anchor))
        distances = sorted(distances, key=lambda item: item[0])
        distances = distances[0:3]
        anchors = [ distance[1] for distance in distances ]
        return anchors

    def heuristic(self, node, anchors_for_node):
        return node.level

class MostRelevantAnchorsSimulator(Simulator):
    def initialize_nodes(self, n, anchor_nodes, length):
        super().initialize_nodes(n, anchor_nodes, length)
        for node in self.nodes:
            if node.level==0:
                node.heuristic=0

    def heuristic(self, node, anchors_for_node):
        return sum( [ anchors.heuristic for anchors in anchors_for_node] ) + 1

    def find_anchors_of(self, node, anchors):
        distances = list()
        for anchor in anchors:
            distance = self.calculate_distance(node.true_position, anchor.true_position)
            if distance > self.radio_range:
                continue
            distances.append((anchor.heuristic, anchor))
        distances = sorted(distances, key=lambda item: item[0])
        distances = distances[0:3]
        anchors = [ distance[1] for distance in distances ]
        return anchors

if __name__ == '__main__':
    errors = list()
    N = 20
    n_anchors = 5
    L = 50
    noise_ratio = 0.3

    non_anchor_nodes= N - n_anchors

    x = list()
    y = list()
    z = list()
    for R in range(8, 25, 2):
        x.append(R)
        localization_ratio = list()
        ale_list = list()
        for i in range(15):
            sim = MostRelevantAnchorsSimulator(N, n_anchors, L, R, noise_ratio)
            sim.run(4)
            loc_ratio, ale = sim.metrics()
            localization_ratio.append(loc_ratio)
            ale_list.append(ale)
        y.append(numpy.mean(localization_ratio))
        z.append(numpy.mean(ale_list))


    pyplot.plot(x, y)
    pyplot.plot(x, z)
    pyplot.show()
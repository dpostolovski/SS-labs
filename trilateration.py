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
            anchors = [item for item in self.nodes if item.level is not None and item.level <= level]
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
        ale=0
        if errors.__len__() !=0:
            ale = sum(errors) / (errors.__len__() * self.radio_range )
        return localized_ratio * 100, ale * 100

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


def metrics_extract(simulator_class, init_params, runparams, iterrations=15):
    localization_ratio = list()
    ale_list = list()
    for i in range(iterrations):
        simulator_object = simulator_class(*init_params)
        simulator_object.run(runparams)
        loc_ratio, ale = simulator_object.metrics()
        localization_ratio.append(loc_ratio)
        ale_list.append(ale)

    return numpy.mean(localization_ratio), numpy.mean(ale_list)

LOCALIZED_LABEL = 'Localized %'
ALE_LABEL = 'ALE %'
NOISE_RATIO_LABEL = 'Localization noise'
ANCHOR_NODES = 'Anchor nodes'
RADIO_RANGE_LABEL = 'Radio range'
def non_iterative_by_R(N, n_anchors, L, noise_ratio):
    x = list()
    y = list()
    z = list()
    for R in range(5, 25, 2):
        x.append(R)
        loc_ratio, ale = metrics_extract(ClosestAnchorsSimulator, init_params=(N, n_anchors, L, R, noise_ratio),
                                         runparams=1)
        y.append(loc_ratio)
        z.append(ale)

    pyplot.plot(x, y,  label=LOCALIZED_LABEL)
    pyplot.plot(x, z, label=ALE_LABEL)
    pyplot.xlabel(RADIO_RANGE_LABEL)
    pyplot.title('Non-iterative, L:{length}, nodes:{nodes}, anchors:{anchors}, noise:{noise}'.format(length=L, nodes=N, anchors=n_anchors, noise=noise_ratio))
    pyplot.legend()
    pyplot.savefig('non_iterative_by_R.png')
    pyplot.show()

def non_iterative_by_R_and_noise(N, n_anchors, L):
    for noise_ratio in range(1, 7):
        noise_ratio = noise_ratio/10.0

        x = list()
        y = list()
        z = list()
        for R in range(5, 25, 2):
            x.append(R)
            loc_ratio, ale = metrics_extract(ClosestAnchorsSimulator, init_params=(N, n_anchors, L, R, noise_ratio),
                                             runparams=1, iterrations=25)
            y.append(loc_ratio)
            z.append(ale)

        #pyplot.plot(x, y,  label=LOCALIZED_LABEL + ", noise:{noise}".format(noise=noise_ratio))
        pyplot.plot(x, z, label=ALE_LABEL + ", noise:{noise}".format(noise=noise_ratio))
    pyplot.xlabel(RADIO_RANGE_LABEL)
    pyplot.title('Non-iterative, L:{length}, nodes:{nodes}, anchors:{anchors}'.format(length=L, nodes=N, anchors=n_anchors, noise=noise_ratio))
    pyplot.legend()
    pyplot.savefig('non_iterative_by_R_and_noise.png')
    pyplot.show()

def non_iterative_by_f(N, L, noise_ratio, R):
    x = list()
    y = list()
    z = list()
    for n_anchors in range(1, N-1, 1):
        x.append(n_anchors)
        loc_ratio, ale = metrics_extract(ClosestAnchorsSimulator, init_params=(N, n_anchors, L, R, noise_ratio),
                                         runparams=1)
        y.append(loc_ratio)
        z.append(ale)

    pyplot.plot(x, y, label=LOCALIZED_LABEL)
    pyplot.plot(x, z, label=ALE_LABEL)
    pyplot.xlabel(ANCHOR_NODES)
    pyplot.title('Non-iterative, L:{length}, nodes:{nodes}, radio range:{range}, noise:{noise}'.format(length=L, nodes=N, range=R,
                                                                                         noise=noise_ratio))
    pyplot.legend()
    pyplot.savefig('non_iterative_by_f.png')
    pyplot.show()

def non_iterative_by_noise(N, n_anchors, L, R):
    x = list()
    y = list()
    z = list()
    for noise_ratio in range(1, 10, 1):
        x.append(noise_ratio/10.0)
        loc_ratio, ale = metrics_extract(ClosestAnchorsSimulator, init_params=(N, n_anchors, L, R, noise_ratio/10.0),
                                         runparams=1, iterrations=40)
        y.append(loc_ratio)
        z.append(ale)

    pyplot.plot(x, z, label=ALE_LABEL)
    pyplot.xlabel(NOISE_RATIO_LABEL)
    pyplot.title('Non-iterative,L:{length}, nodes:{nodes}, radio range:{range}, anchors:{anchors}'.format(length=L,nodes=N, range=R,
                                                                                           anchors=n_anchors))
    pyplot.legend()
    pyplot.savefig('non_iterative_by_noise.png')
    pyplot.show()


def iterative_by_R_and_anchors(N, n_anchors, L, noise_ratio, maxiterations):

    for n_anchors in range(3, 10, 2):
        x = list()
        y = list()
        z = list()
        for R in range(5, 25, 2):
            x.append(R)
            loc_ratio, ale = metrics_extract(ClosestAnchorsSimulator, init_params=(N, n_anchors, L, R, noise_ratio),
                                             runparams=maxiterations)
            y.append(loc_ratio)
            z.append(ale)

        pyplot.plot(x, z, label=ALE_LABEL + ', anchors:{anchors}'.format(anchors=n_anchors))
    pyplot.xlabel(RADIO_RANGE_LABEL)
    pyplot.title('Iterative, max iter.:{maxiterations}, L:{length}, nodes:{nodes}, noise:{noise}'.format(maxiterations=maxiterations,length=L, nodes=N, anchors=n_anchors, noise=noise_ratio))
    pyplot.legend()
    pyplot.savefig('iterative_by_R.png')
    pyplot.show()

def iterative_by_f(N, L, noise_ratio, R,maxiterations):
    x = list()
    y = list()
    z = list()
    for n_anchors in range(1, N-1, 1):
        x.append(n_anchors)
        loc_ratio, ale = metrics_extract(ClosestAnchorsSimulator, init_params=(N, n_anchors, L, R, noise_ratio),
                                         runparams=maxiterations)
        y.append(loc_ratio)
        z.append(ale)

    pyplot.plot(x, z, label=ALE_LABEL)
    pyplot.xlabel(ANCHOR_NODES)
    pyplot.title('Iterative, max iter.:{maxiterations}, L:{length}, nodes:{nodes}, radio range:{range}, noise:{noise}'.format(maxiterations=maxiterations,length=L, nodes=N, range=R,
                                                                                         noise=noise_ratio))
    pyplot.legend()
    pyplot.savefig('iterative_by_f.png')
    pyplot.show()

def iterative_by_noise(N, n_anchors, L, R,maxiterations):
    x = list()
    y = list()
    z = list()
    for noise_ratio in range(1, 10, 1):
        x.append(noise_ratio/10.0)
        loc_ratio, ale = metrics_extract(ClosestAnchorsSimulator, init_params=(N, n_anchors, L, R, noise_ratio/10.0),
                                         runparams=maxiterations, iterrations=40)
        y.append(loc_ratio)
        z.append(ale)

    pyplot.plot(x, z, label=ALE_LABEL)
    pyplot.xlabel(NOISE_RATIO_LABEL)
    pyplot.title('Iterative, max iter.:{maxiterations}, nodes:{nodes}, radio range:{range}, anchors:{anchors}'.format(maxiterations=maxiterations,length=L,nodes=N, range=R,
                                                                                           anchors=n_anchors))
    pyplot.legend()
    pyplot.savefig('iterative_by_noise.png')
    pyplot.show()
if __name__ == '__main__':
    N = 20
    n_anchors = 5
    L = 50
    noise_ratio = 0.3
    R = 15
    maxiterations=10

    non_iterative_by_R(N, n_anchors, L, noise_ratio)
    non_iterative_by_R_and_noise(N, n_anchors, L)
    non_iterative_by_f(N, L, noise_ratio, R)
    non_iterative_by_noise(N, n_anchors, L, R+10)


    iterative_by_R_and_anchors(N, n_anchors, L, noise_ratio,maxiterations)
    iterative_by_f(N, L, noise_ratio, R,maxiterations)
    iterative_by_noise(N, n_anchors, L, R+10,maxiterations)

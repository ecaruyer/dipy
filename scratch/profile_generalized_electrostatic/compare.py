from __future__ import with_statement

import sys

import dipy.core.sphere_stats as sphere_stats
import dipy.core.sphere as sphere
import generalized_electrostatic
from matplotlib import pyplot as plt

import time

mode = None
if len(sys.argv) > 1 and sys.argv[1] == '-s':
    mode = "subdivide"

class Timer(object):
    def __enter__(self):
        self.__start = time.time()

    def __exit__(self, type, value, traceback):
        # Error handling here
        self.__finish = time.time()

    def duration_in_seconds(self):
        return self.__finish - self.__start


def func_minimize_adhoc(init_hemisphere, nb_iterations):
    opt = sphere.disperse_charges(init_hemisphere, nb_iterations)[0]
    return opt.vertices

def func_minimize_scipy(init_pointset, nb_iterations):
    vects = init_pointset.ravel()
    return generalized_electrostatic.disperse_charges(init_pointset, nb_iterations)

nb_points = [20, 40, 60]
nb_subdivide = [2, 3, 4]

if __name__ == '__main__':
    fig = plt.figure()
    for (idx, subplot_index) in enumerate(['131', '132', '133']):
        nb_repetitions = 1
        nb_trials = 3

        execution_time_adhoc = []
        execution_time_scipy = []
        minimum_adhoc = []
        minimum_scipy = []

        if mode == "subdivide":
            init_sphere = sphere.unit_octahedron.subdivide(nb_subdivide[idx])
            init_hemisphere = sphere.HemiSphere.from_sphere(init_sphere)
            init_pointset = init_hemisphere.vertices
        else:
            init_pointset = sphere_stats.random_uniform_on_sphere(nb_points[idx])
            init_hemisphere = sphere.HemiSphere(xyz=init_pointset)
        print "nb_points = %d" % init_pointset.shape[0]

        for j in range(nb_trials):
            print "\tIteration %d out of %d." % (j + 1, nb_trials)

            for nb_iterations in range(12):
                # The time of an iteration of disperse charges is much
                # faster than an iteration of fmin_slsqp.
                nb_iterations_dipy = 20 * nb_iterations

                # Measure execution time for dipy.core.sphere.disperse_charges
                timer = Timer()
                with timer:
                    for i in range(nb_repetitions):
                        opt = func_minimize_adhoc(init_hemisphere, nb_iterations_dipy)
                execution_time_adhoc.append(timer.duration_in_seconds() / nb_repetitions)
                minimum_adhoc.append(generalized_electrostatic.f(opt.ravel()))

                # Measure execution time for generalized_electrostatic.disperse_charges
                timer = Timer()
                with timer:
                    for i in range(nb_repetitions):
                        opt = func_minimize_scipy(init_pointset, nb_iterations)
                execution_time_scipy.append(timer.duration_in_seconds() / nb_repetitions)
                minimum_scipy.append(generalized_electrostatic.f(opt.ravel()))

        ax = fig.add_subplot(subplot_index)
        ax.plot(execution_time_scipy, minimum_scipy, 'g+')
        ax.plot(execution_time_adhoc, minimum_adhoc, 'r+')
        ax.set_yscale('log')
    plt.show()

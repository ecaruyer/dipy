from __future__ import with_statement
import dipy.core.sphere_stats as sphere_stats
import dipy.core.sphere as sphere
import generalized_electrostatic
from matplotlib import pyplot as plt

import time

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
    vects = init_pointset.reshape(nb_points * 3)
    return generalized_electrostatic.disperse_charges(init_pointset, nb_iterations)



if __name__ == '__main__':
    fig = plt.figure()
    for (nb_points, subplot_index) in zip([30, 60, 120], ['131', '132', '133']):
        print "nb_points = %d" % nb_points
        nb_repetitions = 1
        nb_trials = 5

        execution_time_adhoc = []
        execution_time_scipy = []
        minimum_adhoc = []
        minimum_scipy = []
        for j in range(nb_trials):
            print "\tIteration %d out of %d." % (j + 1, nb_trials)
            init_pointset = sphere_stats.random_uniform_on_sphere(n=nb_points)
            init_hemisphere = sphere.HemiSphere(xyz=init_pointset)
        
            for nb_iterations in range(20):
                # The time of an iteration of disperse charges is much
                # faster than an iteration of fmin_slsqp.
                nb_iterations_dipy = 20 * nb_iterations
                
                # Measure execution time for dipy.core.sphere.disperse_charges
                timer = Timer()
                with timer:
                    for i in range(nb_repetitions):
                        opt = func_minimize_adhoc(init_hemisphere, nb_iterations_dipy)
                execution_time_adhoc.append(timer.duration_in_seconds() / nb_repetitions)
                minimum_adhoc.append(generalized_electrostatic.f(opt.reshape(nb_points * 3)))
        
                # Measure execution time for generalized_electrostatic.disperse_charges
                timer = Timer()
                with timer:
                    for i in range(nb_repetitions):
                        opt = func_minimize_scipy(init_pointset, nb_iterations)
                execution_time_scipy.append(timer.duration_in_seconds() / nb_repetitions)
                minimum_scipy.append(generalized_electrostatic.f(opt.reshape(nb_points * 3)))

        ax = fig.add_subplot(subplot_index)
        ax.plot(execution_time_scipy, minimum_scipy, 'g+')
        ax.plot(execution_time_adhoc, minimum_adhoc, 'r+')
        ax.set_yscale('log')
    plt.show()

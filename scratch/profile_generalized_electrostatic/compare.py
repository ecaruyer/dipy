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

execution_time_adhoc = []
execution_time_scipy = []
minimum_adhoc = []
minimum_scipy = []

if __name__ == '__main__':
    nb_points = 30
    nb_repetitions = 10
    nb_trials = 10
    for j in range(nb_trials):
        print "Iteration %d out of %d." % (j + 1, nb_trials)
        init_pointset = sphere_stats.random_uniform_on_sphere(n=nb_points)
        init_hemisphere = sphere.HemiSphere(xyz=init_pointset)
    
        for nb_iterations in range(10):
            nb_iterations_dipy = 10 * nb_iterations
            
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

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(execution_time_scipy, minimum_scipy, 'g+')
ax.plot(execution_time_adhoc, minimum_adhoc, 'r+')
ax.set_yscale('log')
plt.show()

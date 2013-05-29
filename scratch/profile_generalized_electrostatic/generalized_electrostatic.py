from scipy import optimize
import numpy as np
import warnings


def equality_constraints(vects, *args):
    """Spherical equality constraint. Returns 0 if vects lies on the unit sphere.
    
    Parameters
    ----------
    vects : array-like shape (N * 3)
    
    Returns
    -------
    array shape (N,) : Difference between squared vector norms and 1.
    """
    N = vects.shape[0] / 3
    vects = vects.reshape((N, 3))
    return (vects ** 2).sum(1) - 1.0


def grad_equality_constraints(vects, *args):
    """Return normals to the surface constraint (wich corresponds to 
    the gradient of the implicit function).

    Parameters
    ----------
    vects : array-like shape (N * 3)

    Returns
    -------
    array shape (N, N * 3). grad[i, j] contains
    $\partial f_i / \partial x_j$
    """
    N = vects.shape[0] / 3
    vects = vects.reshape((N, 3))
    vects = (vects.T / np.sqrt((vects ** 2).sum(1))).T
    grad = zeros((N, N * 3))
    for i in range(3):
    	grad[:, i * N:(i+1) * N] = np.diag(vects[:, i])
    return grad

	
def f(vects, alpha=2.0, **kwargs):
    """Electrostatic-repulsion objective function. The alpha paramter controls 
    the power repulsion (energy varies as $1 / r^\alpha$).

    Paramters
    ---------
    vects : array-like shape (N * 3,)
    alpha : floating-point. controls the power of the repulsion. Default is 1.0
    weights : array-like, shape (N, N)

    Returns
    -------
    energy : sum of all interactions between any two vectors.
    """
    nb_points = vects.shape[0] / 3
#    weights = kwargs.get('weights', np.ones((nb_points, nb_points)) / nb_points**2)
    weights = kwargs.get('weights', np.ones((nb_points, nb_points)))
    charges = vects.reshape((nb_points, 3))
    all_charges = np.concatenate((charges, -charges))
    all_charges = all_charges[:, None]
    r = charges - all_charges
    r_mag = np.sqrt((r*r).sum(-1))[:, :, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        potential = 1 / r_mag**alpha

    d = np.arange(len(charges))
    potential[d,d] = 0
    potential = potential[:nb_points] + potential[nb_points:]
    potential = weights * potential.sum(-1)
    potential = potential.sum()
    return potential


def grad_f(vects, alpha=2.0, **kwargs):
    """1st-order derivative of electrostatic-like repulsion energy.

    Parameters
    ----------
    vects : array-like shape (N * 3,)
    alpha : floating-point. controls the power of the repulsion. Default is 1.0
    
    Returns
    -------
    grad : gradient of the objective function 
    """
    nb_points = vects.shape[0] / 3
#    weights = kwargs.get('weights', np.ones((nb_points, nb_points)) / nb_points**2)
    weights = kwargs.get('weights', np.ones((nb_points, nb_points)))
    charges = vects.reshape((nb_points, 3))
    all_charges = np.concatenate((charges, -charges))
    all_charges = all_charges[:, None]
    r = charges - all_charges
    r_mag = np.sqrt((r*r).sum(-1))[:, :, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forces = -2 * alpha * r / r_mag**(alpha + 2.)

    d = np.arange(len(charges))
    forces[d,d] = 0
    forces = forces[:nb_points] + forces[nb_points:]
    forces = forces * weights.reshape((nb_points, nb_points, 1))
    forces = forces.sum(0)
    return forces.reshape((nb_points * 3))


def disperse_charges(init_pointset, nb_iter, tol=1.0e-3):
    """Reimplementation of disperse_charges making use of 
    scipy.optimize.fmin_slsqp."""

    K = init_pointset.shape[0]
    vects = optimize.fmin_slsqp(f, init_pointset.reshape(K * 3), 
                                f_eqcons=equality_constraints,
                                fprime=grad_f, iter=nb_iter, acc=tol, 
                                args=[], iprint=0)
    return vects




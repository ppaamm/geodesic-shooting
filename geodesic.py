import numpy as np
from scipy.linalg import sqrtm, expm
import matplotlib as mpl
import matplotlib.pyplot as plt

###############################################################################
###################################  UTILS  ###################################
###############################################################################

def coshm(M):
    return .5 * (expm(M) + expm(-M))

def sinhm(M):
    return .5 * (expm(M) - expm(-M))

def Is_equal(mu0, sigma0, mu1, sigma1):
    return np.array_equal(mu0, mu1) and np.array_equal(sigma0, sigma1)

def is_scalar(M):
    diag = np.diagonal(M)
    if np.count_nonzero(M - np.diag(diag)) > 0:
        return False
    a = diag[0]
    for n in range(diag.shape[0]):
        if diag[n] != a: return False
    return True
    
def sqrtm_2x2(A):
    """
    Square root of 2x2 matrices obtained from "The Square Roots 
    of 2 x 2 Matrices" by D. Sullivan
    """
    sq = np.lib.scimath.sqrt
    # Case 1: A is a scalar matrix
    if is_scalar(A):
        return sq(A[0,0]) * np.identity(2)
    
    # Case 2: A is not a scalar matrix
    D = np.linalg.det(A)
    T = np.trace(A)
    s = sq(D)
    t = sq(T + 2 * s)
    if D == 0 and T == 0:
        raise Exception('Matrix has no square root')
    
    return (1./ t) * (A + s * np.identity(2)) 


###############################################################################
############################  NORMS AND DISTANCES  ############################
###############################################################################

def norm_geo( mu, Sigma, Sigma0, dev=1e-5 ):
    """ Norm of (mu, Sigma) at the point (mu0, Sigma0)
    """
    #Sigma0 = Sigma0 + dev * np.identity(Sigma0.shape[0])
    P = [[], Sigma0]
    V = [mu, Sigma]
    return np.sqrt(scalar_product(P, V, V))


def scalar_product(P, V, W):
    """
    Scalar product of tangent vectors V and W at point P.
    """
    phi = np.linalg.inv(P[1])
    term1 = np.transpose(V[0]).dot(phi).dot(W[0])
    term2 = .5 * np.trace(phi.dot(V[1]).dot(phi).dot(W[1]))
    return term1 + term2


###############################################################################
############################# PARALLEL TRANSPORT  #############################
###############################################################################

def reverse_time(t, backward):
    return (1-t)**backward * t**(1 - backward)

def parallel_transport(W1, mu0, sigma0, mu0_v, sigma0_v, n_steps, 
                                backward=False, accuracy=1e-4):
    """ 
    Numerically integrate the parallel transport equations backwards with RK4
    """
    acc = accuracy
    dt = 1.0 / n_steps
    W = [W1[0], W1[1]]
    for i in range(n_steps):
        t1 = reverse_time(i*dt, backward)
        k1 = deriv(t1, W, acc, mu0, sigma0, mu0_v, sigma0_v)
        t2 = reverse_time(dt*(i) + dt/2, backward)
        k2 = deriv(t2, [W[0]-dt/2*k1[0],W[1]-dt/2*k1[1]], acc, mu0, sigma0, mu0_v, sigma0_v)
        k3 = deriv(t2, [W[0]-dt/2*k2[0],W[1]-dt/2*k2[1]], acc, mu0, sigma0, mu0_v, sigma0_v)
        t4 = reverse_time(dt*(i) + dt, backward)
        k4 = deriv(t4,[W[0]-dt*k3[0],W[1]-dt*k3[1]], acc, mu0, sigma0, mu0_v, sigma0_v)
        W = [W[0] - dt/6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
             W[1] - dt/6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])]
    return W


def deriv( t, y, acc, mu0, sigma0, mu0_t, sigma0_t ):
    """ Integrate the parallel transport equation
    INPUTS: 
    t        : time 
    y        : vector field
    acc      : required accuracy
    mu0      : initial mean
    sigma0   : initial covariance
    mu0_t    : initial mean gradient
    sigma0_t : initial covariance gradient
    OUTPUTS: 
    y1       : the derivative
    """
    [mu_t, sigma_t] = Imai_geo(mu0, sigma0, mu0_t, sigma0_t, t)
    [mu_dt, sigma_dt] = Imai_geo(mu0, sigma0, mu0_t, sigma0_t, t+acc)
    Ddist = [(mu_dt - mu_t)/acc, (sigma_dt - sigma_t)/acc]
    sigma_t_inv = np.linalg.inv(sigma_t)
    y1 = []
    y1.append(.5 * (Ddist[1].dot(sigma_t_inv).dot(y[0]) + y[1].dot(sigma_t_inv).dot(Ddist[0])))
    y_trans = np.transpose(y[0])
    term2 = Ddist[0].dot(y_trans)
    y1.append(y1[0] + .5 * (term2 + np.transpose(term2)))
    
    return y1


###############################################################################
##########################  EXACT GEODESIC SHOOTING  ##########################
###############################################################################

def Imai_geo( mu0, sigma0, mu0_t, sigma0_t, s ):
    """
      Compute the exact geodesic for initial point and initial velocity, given
      formula in "Remarks on geodesics for multivariate normal models" or "An 
      explicit solution of information geodesic equations for the multivariate
      normal model"
    
      INPUT : 
      mu0, sigma0 : Initial points
      mu0_t, sigma0_t : Initial velocities
      s :           Position (instant)
    
      OUTPUT :
      mu, sigma :   Point on the geodesic at time s
    """
    if (np.count_nonzero(mu0_t) == 0 and np.count_nonzero(sigma0_t) == 0):
        mu = mu0
        sigma = sigma0
    else:
        #return mu0 + s * mu0_t, sigma0 + s * sigma0_t
        
        phi0 = np.linalg.inv(sigma0)
        v0 = phi0.dot(mu0)
        phi0_t = - phi0.dot(sigma0_t).dot(phi0)
        v0_t = phi0.dot(mu0_t)
        
        phis, vs = exact_geodesic(v0, phi0, v0_t, phi0_t,s)
        
        sigma = np.linalg.inv(phis)
        mu = sigma.dot(vs)
    return mu, sigma
    #return np.real(mu), np.real(sigma)

def exact_geodesic(v, phi, dv, dphi, s):
    P = sqrtm_2x2(phi)
    invP = np.linalg.inv(P)
    a = invP.dot(dv)
    invPt = np.linalg.inv(np.transpose(P))
    B = - invP.dot(dphi).dot(invPt)
    M = B.dot(B) + 2 * a.dot(np.transpose(a))
    #G = sqrtm(B**2 + 2 * a.dot(np.transpose(a)), disp=False)[0]
    #G = sqrtm(M)
    G = sqrtm_2x2(M)
    #print("G**2", G.dot(G), "M", M)
    Gs2 = .5 * s * G
    shm = sinhm(Gs2)
    g_ginv = np.linalg.pinv(G)
    R = coshm(Gs2) - B.dot(g_ginv).dot(shm)
    phis = P.dot(R).dot(np.transpose(R)).dot(P)
    #phis = P.dot(R).dot(np.transpose(R)).dot(np.transpose(P))
    vs = 2 * P.dot(R).dot(shm).dot(g_ginv).dot(a) + phis.dot(np.linalg.inv(phi)).dot(v)
    return phis, vs



###############################################################################
#############################  GEODESIC SHOOTING  #############################
###############################################################################

def shoot_geo_Imai(mu0, sigma0, mu1, sigma1, 
                   acc=1e-4, epsilon=1e-6, max_iter=1000, n_steps=100, norm_min=.05):
    error = 1
    #mu0_t = mu1 - mu0
    mu0_t = np.zeros(mu0.shape)
    sigma0_t = np.zeros(sigma0.shape)
    P = [mu1, sigma1]
    n = 0
    errors = []
    
    while error >= epsilon:
        if n % 10 == 0:
            print('Iteration ' + str(n))
        # Numerically integrate the geodesic equations for given initial conditions
        [mu_1, sigma_1] = Imai_geo(mu0, sigma0, mu0_t, sigma0_t, 1)
        
        # Calculate error
        W1 = [mu1 - mu_1, sigma1 - sigma_1]
        error = norm_geo(W1[0], W1[1], sigma1)[0]
        errors.append(error)
        #print(error)
        
        # Numerically integrate the parallel transport eqns for final velocities W(1)
        W0 = parallel_transport(W1, mu0, sigma0, mu0_t, sigma0_t, n_steps, 
                                backward=True, accuracy=acc)
        
        # Numerically calculate Jacobi field J(1)
        alpha = epsilon / norm_geo(W0[0], W0[1], sigma0)
        [Ja, Jb] = Imai_geo(mu0, sigma0, mu0_t + alpha*W0[0], sigma0_t + alpha*W0[1], 1)
        J1 = [(Ja - mu_1)/alpha, (Jb - sigma_1)/alpha]
              
        # Determine proper update size
        s1 = scalar_product(P, W1, J1) / norm_geo(J1[0], J1[1],sigma_1)
        norm_W1 = norm_geo(W1[0], W1[1], sigma_1)
        if norm_W1 > norm_min:
            s = norm_min / norm_W1 * s1
        else:
            s = s1
        
        # Update tangent vector
        mu0_t = mu0_t + s * W0[0]
        sigma0_t = sigma0_t + s * W0[1]
        
        n = n + 1
        if n > max_iter:
            return mu0_t, sigma0_t, errors

    return mu0_t, sigma0_t, errors
            

###############################################################################
###################################  PLOT  ####################################
###############################################################################

def add_2d_gaussian_patch(mu, sigma, color):
    chisquare_val = 0.089
    
    v, w = np.linalg.eigh(sigma)
    v = chisquare_val * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    
    # Plot an ellipse to show the Gaussian component
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mu, v[0], v[1], 180. + angle,
                              color=color, fill=False, linestyle='solid')
    return ell

def plot_2d_gaussian(mu, sigma, color):
    #fig = plt.figure(0)
    #ax = fig.add_subplot(111, aspect='equal')
    ell = add_2d_gaussian_patch(mu, sigma, color)
    fig, ax = plt.subplots()
    ax.add_patch(ell)
    ax.set_aspect('equal')
    ax.autoscale()
    plt.show()

def plot_2d_geodesic(P, V, n_steps):
    fig, ax = plt.subplots()
    
    ell_start = add_2d_gaussian_patch(P[0], P[1], 'red')
    ax.add_patch(ell_start)
    
    delta = 1./n_steps
    for i in range(1, n_steps):
        #print(i * delta)
        [mu_i, sigma_i] = Imai_geo(P[0], P[1], V[0], V[1], i * delta)
        ell_i = add_2d_gaussian_patch(mu_i, sigma_i, 'blue')
        ax.add_patch(ell_i)
        
    [mu_1, sigma_1] = Imai_geo(P[0], P[1], V[0], V[1], 1)
    ell_1 = add_2d_gaussian_patch(mu_i, sigma_i, 'red')
    ax.add_patch(ell_1)
    
    ax.set_aspect('equal')
    ax.autoscale()
    plt.show()
    
    

###############################################################################
################################  APPLICATION  ################################
###############################################################################

mu0 = np.transpose(np.array([[0, 0]]))
sigma0 = np.array([[1, 0],[0, .1]])
#phi0 = np.linalg.inv(sigma0)

mu1 = np.transpose(np.array([[0, 5]]))
sigma1 = np.array([[.1,0],[0,1]])

#mu0_t = np.transpose(np.array([[1, 1]]))
#sigma0_t = np.array([[.1,0],[0,1]])
#
#s = 1

#Imai_geo( mu0, sigma0, mu0_t, sigma0_t, s )

#mu0_t, sigma0_t, errors = shoot_geo_Imai(mu0, sigma0, mu1, sigma1, max_iter=20)

#mu0_t = np.transpose(np.array([[0, 0]]))
#sigma0_t = np.zeros((2,2))
#sigma0_t = np.identity(2)

#P = [mu0, sigma0]
#V = [mu0_t, sigma0_t]
#plot_2d_geodesic(P, V, 20)
#scalar_product(P, V, V)

mu0_t, sigma0_t, errors = shoot_geo_Imai(mu0, sigma0, mu1, sigma1, max_iter=100)
P = [mu0, sigma0]
V = [mu0_t, sigma0_t]
plot_2d_geodesic(P, V, 10)

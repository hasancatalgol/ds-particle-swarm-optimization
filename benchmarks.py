import numpy as np

# -------------------------------
# Single-objective benchmark functions
# References:
# https://en.wikipedia.org/wiki/Test_functions_for_optimization
# https://www.sfu.ca/~ssurjano/optimization.html
# -------------------------------

def ackley(x):
    x = np.array(x)
    d = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
        - np.exp(np.sum(np.cos(2*np.pi*x)) / d)
        + 20 + np.e
    )

def beale(x):  # 2D
    x = np.array(x)
    return float(
        (1.5 - x[0] + x[0]*x[1])**2
        + (2.25 - x[0] + x[0]*x[1]**2)**2
        + (2.625 - x[0] + x[0]*x[1]**3)**2
    )

def booth(x):  # 2D
    x = np.array(x)
    return float((x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2)

def bukin_n6(x):  # 2D
    x = np.array(x)
    return float(100 * np.sqrt(np.abs(x[1] - 0.01 * x[0]**2)) + 0.01 * np.abs(x[0] + 10))

def cross_in_tray(x):  # 2D
    x = np.array(x)
    return float(
        -0.0001 * (
            np.abs(
                np.sin(x[0])*np.sin(x[1])
                * np.exp(np.abs(100 - np.sqrt(x[0]**2 + x[1]**2)/np.pi))
            ) + 1
        )**0.1
    )

def easom(x):  # 2D
    x = np.array(x)
    return float(-np.cos(x[0])*np.cos(x[1]) *
                 np.exp(-((x[0]-np.pi)**2 + (x[1]-np.pi)**2)))

def eggholder(x):  # 2D
    x = np.array(x)
    return float(
        -(x[1]+47)*np.sin(np.sqrt(abs(x[0]/2 + (x[1]+47))))
        - x[0]*np.sin(np.sqrt(abs(x[0]-(x[1]+47))))
    )

def goldstein_price(x):  # 2D
    x = np.array(x)
    term1 = 1 + (x[0]+x[1]+1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
    term2 = 30 + (2*x[0]-3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
    return float(term1 * term2)

def griewank(x):
    x = np.array(x)
    return float(
        np.sum(x**2) / 4000
        - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
        + 1
    )

def himmelblau(x):  # 2D
    x = np.array(x)
    return float((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)

def holder_table(x):  # 2D
    x = np.array(x)
    return float(-np.abs(np.sin(x[0])*np.cos(x[1]) *
                         np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2)/np.pi))))

def levy(x):
    x = np.array(x)
    w = 1 + (x - 1) / 4
    return float(
        np.sin(np.pi * w[0])**2
        + np.sum((w[:-1]-1)**2 * (1+10*np.sin(np.pi*w[:-1]+1)**2))
        + (w[-1]-1)**2 * (1+np.sin(2*np.pi*w[-1])**2)
    )

def matyas(x):  # 2D
    x = np.array(x)
    return float(0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1])

def mccormick(x):  # 2D
    x = np.array(x)
    return float(np.sin(x[0]+x[1]) + (x[0]-x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1)

def rastrigin(x):
    x = np.array(x)
    return float(10 * len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)))

def rosenbrock(x):
    x = np.array(x)
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

def schaffer_n2(x):  # 2D
    x = np.array(x)
    num = np.sin(x[0]**2 - x[1]**2)**2 - 0.5
    den = (1 + 0.001*(x[0]**2 + x[1]**2))**2
    return float(0.5 + num/den)

def schaffer_n4(x):  # 2D
    x = np.array(x)
    num = np.cos(np.sin(np.abs(x[0]**2 - x[1]**2)))**2 - 0.5
    den = (1 + 0.001*(x[0]**2 + x[1]**2))**2
    return float(0.5 + num/den)

def schwefel(x):
    x = np.array(x)
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def shekel(x, m=10):  # default 10D
    x = np.array(x)
    # Parameters from literature (Shekel's Foxholes)
    a = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],
                  [3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],
                  [6,2,6,2],[7,3.6,7,3.6]])
    c = np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    total = 0
    for i in range(m):
        total += 1.0 / (c[i] + np.sum((x - a[i])**2))
    return -total  # usually maximization, invert for minimization

def six_hump_camelback(x):  # 2D
    x = np.array(x)
    return float(
        (4 - 2.1*x[0]**2 + (x[0]**4)/3)*x[0]**2
        + x[0]*x[1]
        + (-4 + 4*x[1]**2)*x[1]**2
    )

def sphere(x):
    x = np.array(x)
    return float(np.sum(x**2))

def styblinski_tang(x):
    x = np.array(x)
    return float(0.5*np.sum(x**4 - 16*x**2 + 5*x))

def three_hump_camel(x):  # 2D
    x = np.array(x)
    return float(2*x[0]**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2)

def zakharov(x):
    x = np.array(x)
    i = np.arange(1, len(x)+1)
    return float(np.sum(x**2) + (0.5*np.sum(i*x))**2 + (0.5*np.sum(i*x))**4)


# -------------------------------
# Function registry (name → (func, bounds, dim))
# -------------------------------
FUNCTIONS = {
    "Ackley": (ackley, -5, 5, 10),
    "Beale": (beale, -4.5, 4.5, 2),
    "Booth": (booth, -10, 10, 2),
    "Bukin N.6": (bukin_n6, -15, 15, 2),
    "Cross-in-tray": (cross_in_tray, -10, 10, 2),
    "Easom": (easom, -100, 100, 2),
    "Eggholder": (eggholder, -512, 512, 2),
    "Goldstein-Price": (goldstein_price, -2, 2, 2),
    "Griewank": (griewank, -600, 600, 10),
    "Himmelblau": (himmelblau, -5, 5, 2),
    "Holder Table": (holder_table, -10, 10, 2),
    "Levy": (levy, -10, 10, 10),
    "Matyas": (matyas, -10, 10, 2),
    "McCormick": (mccormick, -2, 4, 2),
    "Rastrigin": (rastrigin, -5, 5, 10),
    "Rosenbrock": (rosenbrock, -5, 5, 10),
    "Schaffer N.2": (schaffer_n2, -100, 100, 2),
    "Schaffer N.4": (schaffer_n4, -100, 100, 2),
    "Schwefel": (schwefel, -500, 500, 10),
    "Shekel": (shekel, 0, 10, 4),
    "Six-hump camelback": (six_hump_camelback, -3, 3, 2),
    "Sphere": (sphere, -5, 5, 10),
    "Styblinski-Tang": (styblinski_tang, -5, 5, 10),
    "Three-hump camel": (three_hump_camel, -5, 5, 2),
    "Zakharov": (zakharov, -5, 10, 10),
}

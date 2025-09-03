import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # prevent Tkinter GUI errors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from random import random
from datetime import datetime
from multiprocessing import Pool


# Import benchmark functions
from benchmarks import FUNCTIONS


# -------------------------------
# Config
# -------------------------------
OUTPUT_DIR = "docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_RUNS = 1
N_PARTICLES = 40
MAX_ITER = 60
W_MAX, W_MIN = 0.9, 0.4
C1 = 1.5
C2 = 1.5
LOG_FREQ = 10
USE_TRAILS = True
USE_PARALLEL = True
USE_HYBRID_GD = True


# -------------------------------
# Timestamp helper
# -------------------------------
def timestamped_name(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}"


# -------------------------------
# Particle Class
# -------------------------------
class Particle:
    def __init__(self, min_x, max_x, dimensions):
        self.min_x = min_x
        self.max_x = max_x
        self.position = self.initialize(dimensions)
        self.velocity = self.initialize(dimensions)
        self.best_position = self.position.copy()
        self.best_value = float("inf")
        self.trail = [self.position.copy()]

    def initialize(self, d):
        return np.array([self.min_x + (self.max_x - self.min_x) * random() for _ in range(d)])

    def move(self):
        self.position += self.velocity
        if np.any(self.position < self.min_x) or np.any(self.position > self.max_x):
            self.best_value = float("inf")
        self.position = np.clip(self.position, self.min_x, self.max_x)
        self.trail.append(self.position.copy())


# -------------------------------
# Gradient Descent refinement
# -------------------------------
def gradient_descent(func, x0, lr=0.01, steps=50):
    x = x0.copy()
    for _ in range(steps):
        grad = numerical_gradient(func, x)
        x -= lr * grad
    return x, func(x)


def numerical_gradient(func, x, eps=1e-6):
    grad = np.zeros_like(x)
    fx = func(x)
    for i in range(len(x)):
        x2 = x.copy()
        x2[i] += eps
        grad[i] = (func(x2) - fx) / eps
    return grad


# -------------------------------
# PSO Class
# -------------------------------
class ParticleSwarmOptimization:
    def __init__(self, func, min_x, max_x, dimensions,
                 n_particles, max_iter, w_max, w_min, c1, c2,
                 log_freq, use_parallel):
        self.func = func
        self.min_x, self.max_x = min_x, max_x
        self.dimensions = dimensions
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2
        self.log_freq = log_freq
        self.use_parallel = use_parallel

        self.particles = [Particle(min_x, max_x, dimensions) for _ in range(n_particles)]
        self.best_position = self.particles[0].position.copy()
        self.best_value = float("inf")
        self.history = []
        self.iteration = 0

    def step(self):
        self.iteration += 1
        w = self.w_max - (self.w_max - self.w_min) * (self.iteration / self.max_iter)

        if self.use_parallel:
            with Pool() as pool:
                fitnesses = pool.map(self.func, [p.position for p in self.particles])
        else:
            fitnesses = [self.func(p.position) for p in self.particles]

        for i, particle in enumerate(self.particles):
            fval = float(fitnesses[i])
            if fval < particle.best_value:
                particle.best_value = fval
                particle.best_position = particle.position.copy()
            if fval < self.best_value:
                self.best_value = fval
                self.best_position = particle.position.copy()

        for particle in self.particles:
            inertia = w * particle.velocity
            cognitive = self.c1 * random() * (particle.best_position - particle.position)
            social = self.c2 * random() * (self.best_position - particle.position)
            particle.velocity = inertia + cognitive + social
            particle.move()

        self.history.append(float(self.best_value))

        if self.iteration % self.log_freq == 0 or self.iteration == self.max_iter:
            pos_str = ", ".join([f"{x:.4f}" for x in np.atleast_1d(self.best_position)])
            val = float(self.best_value)
            print(f"[Iter {self.iteration:3d}/{self.max_iter}] "
                  f"Best Value = {val:.6f} | Best Position = ({pos_str})")


# -------------------------------
# Save Convergence
# -------------------------------
def save_convergence_plot(pso, save_name, func_name, dim):
    plt.figure()
    plt.plot(pso.history, marker="o", markersize=3, linewidth=1.5, color="blue")
    if dim == 2:
        plt.title(f"{func_name} Function (2D) – Convergence")
    else:
        plt.title(f"{func_name} Function – Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Value")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Saved convergence plot to {path}")

COLORMAPS = {
    "Beale": "plasma",
    "Booth": "cividis",
    "Cross-in-tray": "inferno",
    "Easom": "coolwarm",
    "Eggholder": "Spectral",
    "Himmelblau": "viridis",
    "Matyas": "magma",
    "Six-hump camelback": "PuOr",
    "Three-hump camel": "YlGnBu",
}

# -------------------------------
# Animation (2D with Trails)
# -------------------------------
def animate_pso_2d(pso, save_name, func_name):
    x = np.linspace(pso.min_x, pso.max_x, 200)
    y = np.linspace(pso.min_x, pso.max_x, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[pso.func(np.array([xx, yy])) for xx, yy in zip(row_x, row_y)]
                  for row_x, row_y in zip(X, Y)])

    # pick colormap
    cmap = COLORMAPS.get(func_name, "terrain")

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
    plt.colorbar(contour, ax=ax)
    ax.set_title(f"{func_name} Function (2D)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    scat = ax.scatter([], [], c="red", s=40, marker="o", label="Particles", zorder=2)
    best_point = ax.scatter([], [], c="white", s=200, marker="*", edgecolors="black", 
                    label="Global Best", linewidths=1.5, zorder=3)
    ax.legend()
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white",
                   fontsize=10, bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

    def init():
        scat.set_offsets(np.empty((0, 2)))
        best_point.set_offsets(np.empty((0, 2)))
        text.set_text("")
        return scat, best_point, text

    def update(frame):
        pso.step()
        positions = np.array([p.position for p in pso.particles])
        scat.set_offsets(positions)
        best_point.set_offsets(pso.best_position)
        text.set_text(f"Iter {frame+1}/{pso.max_iter}\nBest = {float(pso.best_value):.6f}")

        if USE_TRAILS:
            for p in pso.particles:
                trail = np.array(p.trail)
                ax.plot(trail[:, 0], trail[:, 1], color="red", alpha=0.2, linewidth=0.5)

        return scat, best_point, text

    ani = animation.FuncAnimation(fig, update, frames=pso.max_iter, init_func=init,
                                  blit=True, repeat=False, interval=200)

    gif_path = os.path.join(OUTPUT_DIR, f"{save_name}.gif")
    ani.save(gif_path, writer="pillow")
    print(f"🎥 Saved animation to {gif_path}")
    plt.close(fig)


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    for func_name, (func, min_x, max_x, dim) in FUNCTIONS.items():
        for run in range(N_RUNS):
            print(f"\n🚀 Starting {func_name} Run {run+1}/{N_RUNS}")
            run_name = func_name
            pso = ParticleSwarmOptimization(func=func, min_x=min_x, max_x=max_x,
                                            dimensions=dim, n_particles=N_PARTICLES,
                                            max_iter=MAX_ITER, w_max=W_MAX, w_min=W_MIN,
                                            c1=C1, c2=C2, log_freq=LOG_FREQ,
                                            use_parallel=USE_PARALLEL)

            if dim == 1:
                for _ in range(MAX_ITER):
                    pso.step()
                save_convergence_plot(pso, run_name, func_name, dim)

            elif dim == 2:
                animate_pso_2d(pso, run_name, func_name)
                save_convergence_plot(pso, run_name, func_name, dim)

            else:
                for _ in range(MAX_ITER):
                    pso.step()
                save_convergence_plot(pso, run_name, func_name, dim)

            if USE_HYBRID_GD:
                refined_pos, refined_val = gradient_descent(func, pso.best_position)
                print(f"🔧 Hybrid GD Refinement: {refined_pos} -> {float(refined_val):.6f}")


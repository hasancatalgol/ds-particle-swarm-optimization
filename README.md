# Particle Swarm Optimization (PSO) Benchmark Playground

## 📌 What is Particle Swarm Optimization?

**Particle Swarm Optimization (PSO)** is a population-based optimization algorithm inspired by the collective behavior of birds flocking or fish schooling.

- Each particle represents a **candidate solution** in the search space.
- Particles move with a **velocity** that combines:
  - **Inertia** (previous velocity, exploration),
  - **Cognitive component** (the particle’s own best experience),
  - **Social component** (the swarm’s best-known position).
- Over time, the swarm converges toward promising areas in the landscape.

PSO is widely applied in engineering, machine learning, and scientific optimization because it is simple, flexible, and effective.

---

## ⚙️ Key Parameters

PSO behavior is controlled by three main parameters:

- **Inertia weight ($w$):**  
  Controls how much of the previous velocity is preserved.  
  - High $w$ → encourages **exploration** (particles fly further, search broadly).  
  - Low $w$ → encourages **exploitation** (particles slow down, refine solutions).  
  Often decreased over time (e.g., from $0.9 \to 0.4$).

- **Cognitive coefficient ($c_1$):**  
  Scales how strongly a particle is pulled toward its own **personal best position**.  
  - High $c_1$ → particles act more **individually**, focusing on their own discoveries.  

- **Social coefficient ($c_2$):**  
  Scales how strongly a particle is pulled toward the **global best position** of the swarm.  
  - High $c_2$ → particles act more **cooperatively**, following the swarm leader.  

A good balance between $w$, $c_1$, and $c_2$ is crucial:  
- Too much exploration → slow convergence.  
- Too much exploitation → risk of getting stuck in local minima.  
- Typical defaults: $w \approx 0.7–0.9$, $c_1 \approx 1.5–2.0$, $c_2 \approx 1.5–2.0$.

---

### 📐 Velocity Update Formula

The velocity update rule in PSO is:

$$
v_{i}(t+1) = 
w \cdot v_{i}(t) \;+\;
c_{1} \cdot r_{1} \cdot (p_{i}^{best} - x_{i}(t)) \;+\;
c_{2} \cdot r_{2} \cdot (g^{best} - x_{i}(t))
$$

Where:  
- $v_i(t)$: velocity of particle $i$ at iteration $t$  
- $x_i(t)$: current position of particle $i$  
- $p_i^{best}$: particle’s own best-known position  
- $g^{best}$: global best position in the swarm  
- $r_1, r_2 \sim U(0,1)$: random numbers for stochasticity  

---

### 🎨 Visual Intuition

![PSO Velocity Update](docs/pso_velocity_update.png)

In this diagram, each arrow represents a different influence on a particle’s velocity update:

- **Blue arrow – Inertia ($w \cdot v$):**  
  The particle’s tendency to keep moving in the same direction as before.  
  - High $w$ → particles fly further, maintaining momentum (**exploration**).  
  - Low $w$ → particles slow down, focusing more on refinement (**exploitation**).  

- **Green arrow – Cognitive pull ($c_{1} r_{1} (p^{best} - x)$):**  
  The pull toward the particle’s own best-known position.  
  - Encourages **individual learning**.  
  - Each particle “remembers” where it personally found the best solution so far and is drawn back to it.  

- **Orange arrow – Social pull ($c_{2} r_{2} (g^{best} - x)$):**  
  The pull toward the best-known position in the entire swarm.  
  - Encourages **social learning**.  
  - Particles follow the leader — whichever particle has found the best solution so far.  

- **Red arrow – New velocity:**  
  The final velocity is the **vector sum** of inertia, cognitive pull, and social pull.  
  This is the actual direction and speed the particle will move in the next step.


## 📊 About This Project

This project is a **PSO playground** for studying algorithm behavior across a wide range of **benchmark functions**.  

Features:

- **Configurable swarm parameters** (particles, iterations, inertia, cognitive/social weights).
- **Convergence tracking** with plots saved automatically to `docs/`.
- **2D animations** with contour landscapes and particle trails.
- **Hybrid Gradient Descent refinement** for fine-tuning final solutions.
- **Parallel fitness evaluation** for speed on high-dimensional problems.

---

## 📈 Benchmark Functions

### Ackley Function (Convergence Plot) ([details](https://www.sfu.ca/~ssurjano/ackley.html))

![Ackley Convergence](docs/Ackley.png)

The **Ackley function** is a multimodal test function with many local minima.  
It is designed to challenge global optimization algorithms by forcing them to avoid getting trapped.

---

### Beale Function (2D Animation) ([details](https://www.sfu.ca/~ssurjano/beale.html))

![Beale Animation](docs/Beale.gif)

The **Beale function** is a classic 2D test function with steep valleys and multiple local minima.  
PSO must carefully balance exploration and exploitation to reach the global minimum.

---

### Booth Function (2D Animation) ([details](https://www.sfu.ca/~ssurjano/booth.html))

![Booth Animation](docs/Booth.gif)

The **Booth function** is smooth and convex in 2D.  
It is relatively easy for PSO to solve and converges quickly to the global minimum.

---

### Cross-in-Tray Function (2D Animation) ([details](https://www.sfu.ca/~ssurjano/crossit.html))

![Cross-in-Tray Animation](docs/Cross-in-tray.gif)

The **Cross-in-Tray function** has multiple global minima and sharp peaks.  
It is highly multimodal, making it a difficult test for swarm algorithms.

---

### Easom Function (2D Animation) ([details](https://www.sfu.ca/~ssurjano/easom.html))

![Easom Animation](docs/Easom.gif)

The **Easom function** has a single sharp global minimum surrounded by flat plateaus.  
It is extremely deceptive because most of the search space has nearly constant values.

---

### Eggholder Function (2D Animation) ([details](https://www.sfu.ca/~ssurjano/egg.html))

![Eggholder Animation](docs/Eggholder.gif)

The **Eggholder function** is rugged and highly multimodal.  
Its many steep valleys and ridges make it notoriously difficult for PSO to converge.

---

### Griewank Function (Convergence Plot) ([details](https://www.sfu.ca/~ssurjano/griewank.html))

![Griewank Convergence](docs/Griewank.png)

The **Griewank function** has many regularly distributed local minima.  
Despite this, its global structure allows PSO to steadily find the optimum.

---

### Himmelblau Function (2D Animation) ([details](https://www.sfu.ca/~ssurjano/himmel.html))

![Himmelblau Animation](docs/Himmelblau.gif)

The **Himmelblau function** is a 2D test problem with **four global minima**.  
PSO often converges to different minima depending on initialization.

---

### Levy Function (Convergence Plot) ([details](https://www.sfu.ca/~ssurjano/levy.html))

![Levy Convergence](docs/Levy.png)

The **Levy function** is continuous, multimodal, and highly nonlinear.  
It challenges PSO to explore widely before exploitation.

---

### Matyas Function (2D Animation) ([details](https://www.sfu.ca/~ssurjano/matya.html))

![Matyas Animation](docs/Matyas.gif)

The **Matyas function** is convex and simple in 2D.  
PSO converges easily, making it a baseline benchmark.

---

### Rastrigin Function (Convergence Plot) ([details](https://www.sfu.ca/~ssurjano/rastr.html))

![Rastrigin Convergence](docs/Rastrigin.png)

The **Rastrigin function** is multimodal with a large number of local minima.  
It is widely used to test the exploration ability of PSO.

---

### Rosenbrock Function (Convergence Plot) ([details](https://www.sfu.ca/~ssurjano/rosen.html))

![Rosenbrock Convergence](docs/Rosenbrock.png)

The **Rosenbrock function** has a narrow, curved valley leading to the global minimum.  
It is a difficult test because convergence requires precision along this valley.

---

### Schwefel Function (Convergence Plot) ([details](https://www.sfu.ca/~ssurjano/schwef.html))

![Schwefel Convergence](docs/Schwefel.png)

The **Schwefel function** is deceptive and multimodal.  
Its global minimum is far from the origin, which often misleads algorithms like PSO.

---

### Six-Hump Camelback Function (2D Animation) ([details](https://www.sfu.ca/~ssurjano/camel6.html))

![Six-Hump Camelback Animation](docs/Six-hump%20camelback.gif)

The **Six-Hump Camelback function** has six minima, with two global minima.  
It is one of the most common 2D test functions for PSO and GA.

---

### Sphere Function (Convergence Plot) ([details](https://www.sfu.ca/~ssurjano/spheref.html))

![Sphere Convergence](docs/Sphere.png)

The **Sphere function** is the simplest convex benchmark.  
PSO converges very quickly, making it an easy test case.

---

### Three-Hump Camel Function (2D Animation) ([details](https://www.sfu.ca/~ssurjano/camel3.html))

![Three-Hump Camel Animation](docs/Three-hump%20camel.gif)

The **Three-Hump Camel function** is smooth and polynomial in shape.  
It has a single global minimum and is relatively easy for PSO to solve.

---

### Zakharov Function (Convergence Plot) ([details](https://www.sfu.ca/~ssurjano/zakharov.html))

![Zakharov Convergence](docs/Zakharov.png)

The **Zakharov function** combines linear and quadratic terms, with a curved valley structure.  
It is unimodal but requires careful convergence along the valley floor.

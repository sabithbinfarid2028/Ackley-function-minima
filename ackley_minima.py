"""
Ackley function minimization using Differential Evolution (NumPy only).

f(x) = -a*exp(-b*sqrt(mean(x^2))) - exp(mean(cos(c*x))) + a + e
Global minimum: f(0, ..., 0) = 0, search domain: [-32.768, 32.768]^d
"""

import numpy as np


def ackley(x, a=20.0, b=0.2, c=2 * np.pi):
    x = np.asarray(x, dtype=float)
    d = x.size
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return term1 + term2 + a + np.e


def differential_evolution(func, dim=2, bounds=(-32.768, 32.768), pop_size=50,
                           F=0.8, CR=0.9, max_gen=300, tol=1e-10, seed=None):
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    pop = rng.uniform(lo, hi, (pop_size, dim))
    fit = np.array([func(p) for p in pop])
    history = []

    for gen in range(max_gen):
        for i in range(pop_size):
            # pick 3 distinct individuals different from i
            idx = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            a, b, c = pop[idx]

            mutant = np.clip(a + F * (b - c), lo, hi)

            # binomial crossover
            cross = rng.random(dim) < CR
            cross[rng.integers(dim)] = True
            trial = np.where(cross, mutant, pop[i])

            f_trial = func(trial)
            if f_trial < fit[i]:
                pop[i], fit[i] = trial, f_trial

        best = fit.min()
        history.append(best)
        if best < tol:
            break

    best_idx = fit.argmin()
    return pop[best_idx], fit[best_idx], history


if __name__ == "__main__":
    DIM = 5
    x_best, f_best, hist = differential_evolution(ackley, dim=DIM, seed=42)

    print(f"Dimension        : {DIM}")
    print(f"Generations run  : {len(hist)}")
    print(f"Best x           : {np.round(x_best, 6)}")
    print(f"Best f(x)        : {f_best:.3e}")
    print("True minimum     : f(0,...,0) = 0")

    # Optional convergence plot (needs matplotlib)
    try:
        import matplotlib.pyplot as plt

        plt.semilogy(hist, color="black")
        plt.xlabel("Generation")
        plt.ylabel("Best f(x)")
        plt.title("Ackley - Differential Evolution convergence")
        plt.grid(True, alpha=0.3)
        plt.show()
    except ImportError:
        pass

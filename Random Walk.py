import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# --------------------------------------
# SECTION 1: Parameters & Goal
# --------------------------------------
# This code simulates both a single random walk (1D, symmetric) and many independent walks,
# then compares the path statistics and the final position distribution with the Gaussian law.

N = int(input("Please enter the number of steps N: "))
num_trials = 5000  # Number of independent walks for final position histogram

# --------------------------------------
# SECTION 2: Single Random Walk (Path & Local Time)
# --------------------------------------

steps = np.random.choice([-1, 1], size=N)
x_positions = np.cumsum(np.insert(steps, 0, 0))  # Start from origin (0)

# Count frequency of visiting each position in this walk (local time histogram)
visit_hist = Counter(x_positions)
unique_positions = sorted(visit_hist.keys())
frequencies = [visit_hist[pos] for pos in unique_positions]

# --------------------------------------
# SECTION 3: Many Random Walks - Distribution of Final Position (Central Limit Theorem)
# --------------------------------------

final_positions = []
for _ in range(num_trials):
    steps = np.random.choice([-1, 1], size=N)
    final_positions.append(np.sum(steps))

# Empirical histogram of final positions (distribution after N steps)
final_hist = Counter(final_positions)
final_unique_positions = sorted(final_hist.keys())
final_freqs = np.array([final_hist[pos] for pos in final_unique_positions])
final_freqs = final_freqs / final_freqs.sum()  # Normalize to probability

# Theoretical Gaussian (CLT prediction)
mu = 0
sigma = math.sqrt(N)
x_gauss = np.linspace(min(final_unique_positions), max(final_unique_positions), 200)
gauss_pdf = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-((x_gauss - mu) ** 2) / (2 * sigma ** 2))

# --------------------------------------
# SECTION 4: Visualization
# --------------------------------------

# (A) Random Walk Path
plt.figure(figsize=(10, 4))
plt.plot(range(N + 1), x_positions, label="Random Walk Path", color="blue")
plt.xlabel("Step")
plt.ylabel("Position")
plt.title("1D Random Walk Path (Single Trajectory)")
plt.legend()
plt.grid()
plt.show()

# (B) Histogram of Visits to Each Position (Local Time in One Walk)
plt.figure(figsize=(8, 4))
plt.bar(unique_positions, frequencies, color="orange", label="Visit Count (Local Time)")
plt.xlabel("Position")
plt.ylabel("Visit Frequency (one walk)")
plt.title("Histogram: Number of Visits per Position (Single Walk)")
plt.legend()
plt.grid()
plt.show()

# (C) Histogram of Final Positions (Across Many Independent Walks)
plt.figure(figsize=(8, 5))
plt.bar(final_unique_positions, final_freqs, color="deepskyblue", width=1.0, label="Empirical (Many Walks)")
plt.plot(x_gauss, gauss_pdf, "k--", lw=2, label="Normal (Gaussian) Approx.")
plt.xlabel("Final Position after N steps")
plt.ylabel("Probability")
plt.title(f"Distribution of Final Positions after N={N} Steps\n(Random Walk vs. Gaussian)")
plt.legend()
plt.grid()
plt.show()

# --------------------------------------
# PHYSICAL AND COMPUTATIONAL NOTES
# --------------------------------------
# - In a **single walk**, the histogram shows local time (visits), NOT the final distribution!
# - Only after simulating many independent walks and plotting the histogram of *final positions*,
#   the distribution becomes Gaussian.
# - This is the true meaning of random walk's connection to diffusion and Brownian motion.

# - The code visualizes both: (A) path of a single walk, (B) visit histogram (one walk),
#   (C) final position distribution over many walks and its Gaussian approximation.


import numpy as np
import matplotlib.pyplot as plt
import pymatching
import pandas as pd
from scipy.stats import lognorm
from surface_rotated_d3 import build_surface_code

def collect_data():
    points = np.logspace(-4, -1, 10)
    rounds = [3, 5, 10]
    num_simulations = 100
    shots = 10000
    
    data = {"rounds": [], "p": [], "gain": []}
    
    for r in rounds:
        print(f"Processing Round T={r}...")
        for p in points:
            p_phys_total = 1 - (1 - p)**r
            
            # compile once per (r, p)
            c = build_surface_code(rounds=r, p=p)
            error_model = c.detector_error_model(decompose_errors=True)
            matching = pymatching.Matching.from_detector_error_model(error_model)
            sampler = c.compile_detector_sampler()
            
            for _ in range(num_simulations):
                syn, obs = sampler.sample(shots=shots, separate_observables=True)
                pred = matching.decode_batch(syn)
                errors = np.sum(np.any(pred != obs, axis=1))
                
                P_L = errors / shots
                if P_L == 0:
                    P_L = 1.0 / (shots * 10)
                    
                gain = p_phys_total / P_L
                data["rounds"].append(r)
                data["p"].append(p)
                data["gain"].append(gain)
                
    return data

if __name__ == '__main__':
    data = collect_data()
    df = pd.DataFrame(data)
    points = np.logspace(-4, -1, 10)
    rounds = [3, 5, 10]
    
    # export metrics to CV
    summary = df.groupby(["rounds", "p"]).agg(
        mean_gain=("gain", "mean"),
        std_dev_gain=("gain", "std"),
        min_gain=("gain", "min"),
        max_gain=("gain", "max")
    ).reset_index()
    
    summary.to_csv("monte_carlo_results.csv", index=False)
    print("Exported monte_carlo_results.csv")
    
    # visual 1: reliability gain
    plt.figure(figsize=(10, 6))
    colors = {3: 'blue', 5: 'orange', 10: 'green'}
    
    for r in rounds:
        sub = summary[summary["rounds"] == r]
        plt.errorbar(sub["p"], sub["mean_gain"], yerr=sub["std_dev_gain"], fmt='o-', color=colors[r], label=f'T={r}')
        
    plt.xscale('log')
    plt.yscale('log')
    plt.axhline(1, color='red', linestyle='--', label='G=1 (No Advantage)')
    
    # shade the advantage space
    upper_bound_visually = summary['mean_gain'].max() * 5
    plt.fill_between([min(points), max(points)], 1, upper_bound_visually, color='green', alpha=0.1, label='QEC Advantage Zone')
    
    plt.ylim(top=upper_bound_visually)
    plt.xlabel("Physical Error Rate (p)")
    plt.ylabel("Mean Numerical Gain (G)")
    plt.title("Reliability Gain vs Physical Error (100 runs per pt, 10k shots)")
    plt.legend()
    plt.grid(True, which="both", ls='-', alpha=0.2)
    plt.savefig("results/reliability_gain.png")
    plt.close()
    print("Exported reliability_gain.png")
    
    # visual 2: log gittings + threshold approx p=0.001
    focus_ps = [1e-4, 1e-3, 1e-2, 1e-1]
    
    fig, axes = plt.subplots(len(focus_ps), 3, figsize=(15, 12))
    
    for row_idx, focus_p in enumerate(focus_ps):
        # nearest p computation from the logspace
        closest_p = points[np.abs(points - focus_p).argmin()]
        
        for col_idx, r in enumerate(rounds):
            sub = df[(df["rounds"] == r) & (np.isclose(df["p"], closest_p))]
            gains = sub["gain"].values
            
            ax = axes[row_idx, col_idx]
            ax.hist(gains, bins=15, density=True, alpha=0.6, color=colors[r])
            
            # log extraction module
            if len(gains) > 0 and np.all(gains > 0):
                shape, loc, scale = lognorm.fit(gains, floc=0)
                x_val = np.linspace(min(gains)*0.8, max(gains)*1.2, 100)
                pdf = lognorm.pdf(x_val, shape, loc=loc, scale=scale)
                ax.plot(x_val, pdf, 'r-', linewidth=2)
                
            ax.set_title(f"T={r} at p ~ {closest_p:.1e}")
            ax.set_xlabel("Gain")
            ax.set_ylabel("Density")
            if len(ax.get_legend_handles_labels()[0]) > 0:
                ax.legend()
            
    plt.tight_layout()
    # verify if directory results/ exists, otherwise just save to current
    import os
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/gain_distribution_histograms_grid.png")
    plt.close()
    print("Exported gain_distribution_histograms_grid.png")
    print("All tasks completed.")

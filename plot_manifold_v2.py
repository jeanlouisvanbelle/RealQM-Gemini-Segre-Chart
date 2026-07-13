import pandas as pd
import matplotlib.pyplot as plt

def generate_v2_plots():
    print("="*75)
    print("[RUNNING] REALQM STUDIO: PLOTTING HIGH-ITERATION MICRO-SWEEP (Z=30-94)")
    print("="*75)
    
    csv_file = "relaxed_telemetry_Z_30_94_v2.csv"
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[ERROR] Could not read file: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Panel 1: Spectral Graph Radius Scaling
    ax1.scatter(df['Z'], df['Graph_Radius'], color='#1F4E79', s=25, alpha=0.6, label='Isotope Tracking')
    ax1.set_title("RealQM v2.0 Parallel Sweep: High-Iteration Manifold (Z=30 to 94)\n[500 Iterations Un-Capped Core Relaxation]", fontweight='bold', fontsize=13, color='#1F4E79')
    ax1.set_ylabel("Spectral Graph Radius (Size)", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper left')

    # Panel 2: High-Iteration Relaxed Fiedler Stiffness Stability Plateau
    ax2.plot(df['Z'], df['Relaxed_Fiedler_Stiffness'], color='darkred', linestyle='none', marker='o', alpha=0.7, label='Relaxed Network Rigidity')
    
    # Highlight Mercury and Lead Anchors
    hg_peak = df[df['Z'] == 80]
    if not hg_peak.empty:
        ax2.scatter(hg_peak['Z'].iloc[0], hg_peak['Relaxed_Fiedler_Stiffness'].max(), color='gold', s=120, edgecolors='black', zorder=5, label='Mercury Peak (Z=80)')

    ax2.set_xlabel("Atomic Number (Proton Count Z)", fontsize=11)
    ax2.set_ylabel("Relaxed Fiedler Stiffness ($\lambda_1$)", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    output_png = "relaxed_stability_manifold_v2.png"
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] High-resolution graphic compiled as '{output_png}'!")
    print("="*75)

if __name__ == "__main__":
    generate_v2_plots()

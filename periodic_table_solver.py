import numpy as np
import scipy.linalg as la

def generate_alpha_offsets():
    """Tetrahedral loop center offsets for an individual alpha block."""
    d = 0.8415 / np.sqrt(3)
    return np.array([[d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]])

def build_universal_matrix_coordinates(Z, A):
    """
    RealQM Unified Matrix Initialization Rule.
    Factorizes any arbitrary nuclide into concentric shells & polar bipyramids.
    """
    n_alpha = Z // 2
    n_excess = A - 2 * n_alpha
    
    n_anchor = min(n_alpha, 41)
    n_polar = max(0, n_alpha - 41)
    n_buffer = min(n_excess, 36)
    n_fringe = max(0, n_excess - 36)
    
    coords = []
    loop_types = []
    alpha_offsets = generate_alpha_offsets()
    
    # 1. POPULATE ANCHOR CORE SHELLS (Max 41 Alphas)
    # Generates standard nested shell distributions based on capacity thresholds
    for i in range(n_anchor):
        if i == 0:
            center = [0.0, 0.0, 0.0]
        elif i <= 12:  # Inner icosahedral shell horizon
            angle = i * (2 * np.pi / 12)
            center = [2.1 * np.cos(angle), 2.1 * np.sin(angle), 0.2 * (i % 2 - 0.5)]
        else:          # Outer core bounding shell
            angle = i * (2 * np.pi / (n_anchor - 13))
            center = [3.5 * np.cos(angle), 3.5 * np.sin(angle), -0.5 if i % 2 == 0 else 0.5]
            
        for idx, offset in enumerate(alpha_offsets):
            coords.append(np.array(center) + offset)
            loop_types.append('p' if idx % 2 == 0 else 'n')
            
    # 2. POPULATE POLAR CAPS REMAINING ALPHAS
    if n_polar > 0:
        z_apex = 5.0
        for i in range(n_polar):
            sign = 1.0 if i % 2 == 0 else -1.0
            r_offset = 1.1 * (i // 2)
            center = [r_offset, 0.0, sign * z_apex]
            for idx, offset in enumerate(alpha_offsets):
                coords.append(np.array(center) + offset)
                loop_types.append('p' if idx % 2 == 0 else 'n')
                
    # 3. POPULATE INTERMEDIATE NEUTRON BUFFERS (Radius = 2.7 fm)
    if n_buffer > 0:
        for i in range(n_buffer):
            angle = i * (2 * np.pi / n_buffer)
            coords.append([2.7 * np.cos(angle), 2.7 * np.sin(angle), 0.1 * (i % 2)])
            loop_types.append('n')
            
    # 4. POPULATE OUTER FRINGE VALENCE SATELLITES (Radius = 6.8 fm)
    if n_fringe > 0:
        for i in range(n_fringe):
            angle = i * (2 * np.pi / n_fringe)
            coords.append([6.8 * np.cos(angle), 6.8 * np.sin(angle), -0.2 if i % 2 == 0 else 0.2])
            loop_types.append('n')
            
    return np.array(coords), loop_types

def compute_matrix_spectral_radius(coords, loop_types):
    """Evaluates the discrete Laplacian graph rigidity from the baseline interaction array."""
    A_size = len(coords)
    lambda_c = 0.2103
    
    # Fast vectorized broadcasting array calculation maps
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r_matrix = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(r_matrix, 0.1) # Avoid divide-by-zero bounds
    
    # Fermi-Dirac contact horizon truncation operator
    f_cut = 1.0 / (1.0 + np.exp((r_matrix - 2.4) / 0.12))
    
    # Relativistic phase-retardation operator
    phase_mod = np.cos(r_matrix / lambda_c)
    
    # Total magnetic potential matrix layout
    H = -12.5 / r_matrix * f_cut * phase_mod
    
    # Build Graph Laplacian spectrum
    Adjacency = np.abs(np.minimum(H, 0.0))
    np.fill_diagonal(Adjacency, 0.0)
    Degree = np.diag(np.sum(Adjacency, axis=1))
    Laplacian = Degree - Adjacency
    
    eigenvalues = la.eigvalsh(Laplacian)
    return eigenvalues[-1], eigenvalues[1] # Return Radius and Fiedler

def execute_periodic_table_sweep():
    print("="*80)
    print("SUNDANCE V5.4: UNIVERSAL SEGRÈ VALLEY ALGEBRAIC SOLVER RUN")
    print("="*80)
    print(f"{'Nuclide':<12}{'Protons (Z)':<14}{'Mass (A)':<12}{'Graph Radius':<16}{'Fiedler stiffness'}")
    print("-"*80)
    
    # Unified test list looping across light, medium, and heavy structural milestones
    test_elements = [
        {"name": "14-Si-28", "Z": 14, "A": 28},
        {"name": "20-Ca-40", "Z": 20, "A": 40},
        {"name": "26-Fe-56", "Z": 26, "A": 56},
        {"name": "92-U-230", "Z": 92, "A": 230},
        {"name": "92-U-238", "Z": 92, "A": 238}
    ]
    
    for element in test_elements:
        coords, loop_types = build_universal_matrix_coordinates(element["Z"], element["A"])
        graph_radius, fiedler = compute_matrix_spectral_radius(coords, loop_types)
        print(f"{element['name']:<12}{element['Z']:<14}{element['A']:<12}{graph_radius:<16.4f}{fiedler:.4f}")
        
    print("="*80)

if __name__ == "__main__":
    execute_periodic_table_sweep()

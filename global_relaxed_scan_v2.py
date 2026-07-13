import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import multiprocessing as mp
import os
import time

def generate_alpha_offsets():
    """Tetrahedral loop center offsets for an individual alpha block."""
    d = 0.8415 / np.sqrt(3)
    return np.array([[d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]])

def build_universal_matrix_coordinates(Z, A):
    """RealQM Nested Core-Satellite Hierarchy Rule."""
    n_alpha = Z // 2
    n_excess = A - 2 * n_alpha
    
    n_anchor = min(n_alpha, 41)
    n_polar = max(0, n_alpha - 41)
    n_buffer = min(n_excess, 36)
    n_fringe = max(0, n_excess - 36)
    
    coords = []
    loop_types = []
    alpha_offsets = generate_alpha_offsets()
    
    # 1. Anchor Core Shells
    for i in range(n_anchor):
        if i == 0:
            center = [0.0, 0.0, 0.0]
        elif i <= 12:
            angle = i * (2 * np.pi / 12)
            center = [2.1 * np.cos(angle), 2.1 * np.sin(angle), 0.2 * (i % 2 - 0.5)]
        else:
            angle = i * (2 * np.pi / (n_anchor - 13))
            center = [3.5 * np.cos(angle), 3.5 * np.sin(angle), -0.5 if i % 2 == 0 else 0.5]
            
        for idx, offset in enumerate(alpha_offsets):
            coords.append(np.array(center) + offset)
            loop_types.append('p' if idx % 2 == 0 else 'n')
            
    # 2. Polar Caps
    if n_polar > 0:
        z_apex = 5.0
        for i in range(n_polar):
            sign = 1.0 if i % 2 == 0 else -1.0
            r_offset = 1.1 * (i // 2)
            center = [r_offset, 0.0, sign * z_apex]
            for idx, offset in enumerate(alpha_offsets):
                coords.append(np.array(center) + offset)
                loop_types.append('p' if idx % 2 == 0 else 'n')
                
    # 3. Intermediate Buffers
    if n_buffer > 0:
        for i in range(n_buffer):
            angle = i * (2 * np.pi / n_buffer)
            coords.append([2.7 * np.cos(angle), 2.7 * np.sin(angle), 0.1 * (i % 2)])
            loop_types.append('n')
            
    # 4. Outer Fringe Satellites
    if n_fringe > 0:
        for i in range(n_fringe):
            angle = i * (2 * np.pi / n_fringe)
            coords.append([6.8 * np.cos(angle), 6.8 * np.sin(angle), -0.2 if i % 2 == 0 else 0.2])
            loop_types.append('n')
            
    if len(coords) == 0 and Z == 1:
        coords.append([0.0, 0.0, 0.0])
        loop_types.append('p')
            
    return np.array(coords), loop_types

def compute_relaxed_matrix(coords, loop_types, angles):
    """Broadcasting matrix builder tracking orientation transformations."""
    A_size = len(coords)
    lambda_c = 0.2103
    
    thetas = angles[0::2]
    phis = angles[1::2]
    orientations = np.stack([np.sin(thetas)*np.cos(phis), np.sin(thetas)*np.sin(phis), np.cos(thetas)], axis=-1)
    
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r_matrix = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(r_matrix, 0.1)
    
    f_cut = 1.0 / (1.0 + np.exp((r_matrix - 2.4) / 0.12))
    angular_alignment = np.dot(orientations, orientations.T)
    phase_mod = np.cos(r_matrix / lambda_c)
    
    H = -12.5 / r_matrix * f_cut * phase_mod * angular_alignment
    return H

def relaxation_objective(angles, coords, loop_types):
    """Target function driving the L-BFGS-B gradient minimization valley."""
    H = compute_relaxed_matrix(coords, loop_types, angles)
    Adjacency = np.abs(np.minimum(H, 0.0))
    np.fill_diagonal(Adjacency, 0.0)
    Degree = np.diag(np.sum(Adjacency, axis=1))
    Laplacian = Degree - Adjacency
    try:
        eigenvalues = la.eigvalsh(Laplacian)
        return -eigenvalues[-1]
    except:
        return 0.0

def evaluate_single_isotope(args):
    """Worker process mapping out a single unconstrained relaxation pass."""
    Z, A, max_iter = args
    coords, loop_types = build_universal_matrix_coordinates(Z, A)
    A_size = len(coords)
    
    if A_size <= 1:
        return Z, A, 0.0, 0.0
        
    np.random.seed(42)
    initial_angles = np.random.uniform(0.01, 0.1, 2 * A_size)
    bounds = [(0, np.pi), (0, 2*np.pi)] * A_size
    
    res = minimize(relaxation_objective, initial_angles, args=(coords, loop_types),
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
    
    H_opt = compute_relaxed_matrix(coords, loop_types, res.x)
    Adjacency = np.abs(np.minimum(H_opt, 0.0))
    np.fill_diagonal(Adjacency, 0.0)
    Degree = np.diag(np.sum(Adjacency, axis=1))
    Laplacian = Degree - Adjacency
    
    eigenvalues = la.eigvalsh(Laplacian)
    fiedler_val = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    return Z, A, eigenvalues[-1], fiedler_val

def run_parallel_micro_sweep(z_start, z_end, max_iter=500):
    print("="*80)
    print(f"REALQM v2.0 ENGINE: HIGH-ITERATION PARALLEL MICRO-SWEEP (Z={z_start} TO Z={z_end})")
    print(f"CORE COUNT DETECTED: {mp.cpu_count()} WORKERS ON SUNDANCE")
    print("="*80)
    
    # 1. Compile parallel execution queue list
    tasks = []
    for Z in range(z_start, z_end + 1):
        stable_A = int(2 * Z + 0.0075 * (Z ** 2))
        for A in range(max(Z, stable_A - 1), stable_A + 2):
            tasks.append((Z, A, max_iter))
            
    output_file = f"relaxed_telemetry_Z_{z_start}_{z_end}_v2.csv"
    
    # 2. Fire up the multiprocessing pool across all Ryzen cores
    start_time = time.time()
    with open(output_file, "w") as f:
        f.write("Z,A,Graph_Radius,Relaxed_Fiedler_Stiffness\n")
        
        pool = mp.Pool(processes=mp.cpu_count())
        
        # Stream results asynchronously as tasks finish calculation
        for count, result in enumerate(pool.imap_unordered(evaluate_single_isotope, tasks), 1):
            Z, A, radius, fiedler = result
            f.write(f"{Z},{A},{radius:.6f},{fiedler:.6f}\n")
            
            # LIVE IO FLUSH: Push rows straight to disk row-by-row
            f.flush()
            
            if count % 5 == 0 or count == len(tasks):
                print(f"[PROGRESS] Completed {count}/{len(tasks)} target isotope relaxation passes...")
                
        pool.close()
        pool.join()
        
    duration = time.time() - start_time
    print("-"*80)
    print(f"[SUCCESS] Parallelized micro-sweep complete in {duration:.2f} seconds!")
    print(f"-> Telemetry database securely stored in '{output_file}'")
    print("="*80)

if __name__ == "__main__":
    # NEW TARGET: Capture the entire mid-heavy anomaly and the actinide cliff
    # This runs a deep optimization sweep (max_iter=500) from Zinc (Z=30) up to Plutonium (Z=94)
    run_parallel_micro_sweep(z_start=30, z_end=94, max_iter=500)


import numpy as np
import os
from .config import *
from ..progress import ProgressBar, track
from .elastic_tools import (
    read_matrix,
    identify_symmetry,
    stability_criteria,
    vrh_averages,
    anisotropy_indices_from_vrh,
    compliance_matrix,
    youngs_modulus_directional,
    linear_compressibility_directional,
    bulk_modulus_vrh,
    bulk_modulus_directional,
    shear_modulus_vrh,
    plot_2d_polar,
    plot_3d_surface,
    plot_crystallographic_sections,
    save_xyz_grid, save_matrix, save_extremal_values,
    report,
    suggest_matrix_correction,
    swap_c44_c66,
    force_symmetry,
    calculate_thermodynamic_properties,
    calculate_sound_velocities,
    apply_isotropic_stress,
    calculate_polycrystalline_moduli,
    calculate_polycrystalline_anisotropy
)

# Property names and units for output formatting
PROPERTY_NAMES = {
    "E": "Young's Modulus",
    "L": "Linear Compressibility",
    "G": "Shear Modulus",
    "nu": "Poisson's Ratio",
    "B": "Bulk Modulus"
}

PROPERTY_UNITS = {
    "E": "GPa",
    "L": "GPa^-1",
    "G": "GPa",
    "nu": "",
    "B": "GPa"
}

def _save_2d_polar_data(path, prop_type, plane, theta, values, values_min=None):
    """导出 2D 极图曲线数据 (Origin 可直接读: 角度 + 数值 + 笛卡尔坐标)。"""
    import numpy as _np
    theta = _np.asarray(theta)
    values = _np.asarray(values)
    deg = _np.degrees(theta)
    unit = PROPERTY_UNITS.get(prop_type, "")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# MHEC 2D directional {PROPERTY_NAMES.get(prop_type, prop_type)} "
                f"in {plane.upper()} plane (unit: {unit or 'dimensionless'})\n")
        if values_min is not None:
            vmin = _np.asarray(values_min)
            f.write("angle_deg\tangle_rad\tmax\tmin\tx_max\ty_max\tx_min\ty_min\n")
            for k in range(len(theta)):
                xmax, ymax = values[k]*_np.cos(theta[k]), values[k]*_np.sin(theta[k])
                xmin, ymin = vmin[k]*_np.cos(theta[k]), vmin[k]*_np.sin(theta[k])
                f.write(f"{deg[k]:.4f}\t{theta[k]:.6f}\t{values[k]:.6e}\t{vmin[k]:.6e}\t"
                        f"{xmax:.6e}\t{ymax:.6e}\t{xmin:.6e}\t{ymin:.6e}\n")
        else:
            f.write("angle_deg\tangle_rad\tvalue\tx\ty\n")
            for k in range(len(theta)):
                x, y = values[k]*_np.cos(theta[k]), values[k]*_np.sin(theta[k])
                f.write(f"{deg[k]:.4f}\t{theta[k]:.6f}\t{values[k]:.6e}\t{x:.6e}\t{y:.6e}\n")


def main(input_path: str, output_dir: str, auto_mode: bool = True):
    """Main workflow: read, analyze, plot, and export elastic properties."""
    print(f"[INFO] Ensuring output directory \'{output_dir}\' exists...")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory \'{output_dir}\' is ready.")

    # 1. Read and perform basic analysis
    print(f"[INFO] Attempting to read stiffness matrix from: {input_path}")
    C_original = read_matrix(input_path)
    C = C_original.copy()
    print(f"[SUCCESS] Stiffness matrix successfully read.\nMatrix:\n{C_original}")
    
    # Save original matrix before any correction
    original_matrix_path = os.path.join(output_dir, "correction_before.txt")
    np.savetxt(original_matrix_path, C_original)
    print(f"[INFO] Original matrix saved to: {original_matrix_path}")

    # Initial symmetry identification
    print("[INFO] Identifying initial crystal symmetry...")
    sym_initial = identify_symmetry(C)
    print(f"[SUCCESS] Initial crystal symmetry identified: {sym_initial.system}")
    print("[DEBUG] Initial symmetry criteria:\n" + "\n".join([f"  {k}: {v}" for k, v in sym_initial.criteria.items()]))

    # Suggest correction if needed
    suggest_swap, reason = suggest_matrix_correction(C, sym_initial.system)
    if suggest_swap:
        print(f"\n[WARNING] Potential matrix correction suggested: {reason}")
        if auto_mode:
            if AUTO_APPLY_CORRECTION:
                print("[INFO] Auto mode: Applying suggested correction...")
                C = swap_c44_c66(C)
            else:
                print("[INFO] Auto mode: Correction suggested but disabled in config...")
            print("Matrix after correction:\n", C)
            sym_corrected = identify_symmetry(C)
            print(f"[SUCCESS] Crystal symmetry after correction: {sym_corrected.system}")
            # Save corrected matrix
            corrected_matrix_path = os.path.join(output_dir, "correction_after.txt")
            np.savetxt(corrected_matrix_path, C)
            print(f"[INFO] Corrected matrix saved to: {corrected_matrix_path}")
        else:
            # Interactive prompt for user confirmation
            while True:
                user_input = input("Do you want to apply this correction? (y/n): ").lower()
                if user_input == 'y':
                    print("[INFO] Applying C44-C66 swap...")
                    C = swap_c44_c66(C)
                    print("Matrix after correction:\n", C)
                    sym_corrected = identify_symmetry(C)
                    print(f"[SUCCESS] Crystal symmetry after correction: {sym_corrected.system}")
                    # Save corrected matrix
                    corrected_matrix_path = os.path.join(output_dir, "correction_after.txt")
                    np.savetxt(corrected_matrix_path, C)
                    print(f"[INFO] Corrected matrix saved to: {corrected_matrix_path}")
                    break
                elif user_input == 'n':
                    print("[INFO] Correction not applied by user.")
                    break
                else:
                    print("[ERROR] Invalid input. Please enter 'y' or 'n'.")
    else:
        print("[INFO] No matrix correction suggested based on current criteria.")

    # Force symmetry if desired (new feature)
    if auto_mode:
        if FORCE_SYMMETRY:
            print(f"[INFO] Auto mode: Forcing matrix to {sym_initial.system} symmetry...")
            C = force_symmetry(C, sym_initial.system)
            print("Matrix after forcing symmetry:\n", C)
            forced_matrix_path = os.path.join(output_dir, "forced_symmetry_matrix.txt")
            np.savetxt(forced_matrix_path, C)
            print(f"[INFO] Forced symmetry matrix saved to: {forced_matrix_path}")
        else:
            print(f"[INFO] Auto mode: Not forcing symmetry (disabled in config).")
    else:
        print("\n[INPUT] Do you want to force the matrix to conform to the identified symmetry? (y/n): ")
        while True:
            user_input = input().lower()
            if user_input == 'y':
                print(f"[INFO] Forcing matrix to {sym_initial.system} symmetry...")
                C = force_symmetry(C, sym_initial.system)
                print("Matrix after forcing symmetry:\n", C)
                forced_matrix_path = os.path.join(output_dir, "forced_symmetry_matrix.txt")
                np.savetxt(forced_matrix_path, C)
                print(f"[INFO] Forced symmetry matrix saved to: {forced_matrix_path}")
                break
            elif user_input == 'n':
                print("[INFO] Not forcing symmetry.")
                break
            else:
                print("[ERROR] Invalid input. Please enter 'y' or 'n'.")

    print("[INFO] Calculating compliance matrix...")
    S = compliance_matrix(C)
    print("[SUCCESS] Compliance matrix calculated.\nMatrix:\n", S)
    # S4 is not used directly in the current implementation of directional properties, 
    # as youngs_modulus_directional and linear_compressibility_directional now take the 6x6 S matrix.
    # print("[INFO] Converting compliance matrix to 4th order tensor...")
    # S4 = voigt_compliance_to_tensor(S)
    # print("[SUCCESS] 4th order tensor created.")

    print("[INFO] Re-identifying symmetry after potential correction and forcing...")
    sym = identify_symmetry(C) # Re-identify symmetry after potential correction and forcing
    print(f"[SUCCESS] Final crystal symmetry identified: {sym.system}")
    print("[INFO] Checking mechanical stability...")
    stab = stability_criteria(C, sym.system)
    print(f"[SUCCESS] Mechanical stability check completed. Stable: {stab.stable}")
    print("[INFO] Calculating Voigt-Reuss-Hill averages...")
    vrh = vrh_averages(C)
    print("[SUCCESS] VRH averages calculated.")
    print("[INFO] Calculating anisotropy indices...")
    anisotropy = anisotropy_indices_from_vrh(vrh)
    print("[SUCCESS] Anisotropy indices calculated.")

    print("[INFO] Calculating polycrystalline moduli...")
    poly_moduli = calculate_polycrystalline_moduli(C)
    print("[SUCCESS] Polycrystalline moduli calculated.")
    print("[INFO] Calculating polycrystalline anisotropy indices...")
    poly_anisotropy = calculate_polycrystalline_anisotropy(poly_moduli)
    print("[SUCCESS] Polycrystalline anisotropy indices calculated.")

    # Get user input for density and molar mass for thermodynamic properties
    if auto_mode:
        density = DENSITY
        molar_mass = MOLAR_MASS
        print(f"[INFO] Auto mode: Using density = {density} g/cm^3 and molar mass = {molar_mass} g/mol")
    else:
        density = None
        molar_mass = None
        while density is None:
            try:
                density_str = input("\n[INPUT] Enter material density in g/cm^3 (e.g., 5.0): ")
                density = float(density_str)
            except ValueError:
                print("[ERROR] Invalid input. Please enter a number.")
        while molar_mass is None:
            try:
                molar_mass_str = input("[INPUT] Enter material molar mass in g/mol (e.g., 50.0): ")
                molar_mass = float(molar_mass_str)
            except ValueError:
                print("[ERROR] Invalid input. Please enter a number.")

    print("[INFO] Calculating thermodynamic properties...")
    thermo_props = calculate_thermodynamic_properties(C, density, molar_mass)
    print("[SUCCESS] Thermodynamic properties calculated.")

    print("[INFO] Calculating sound velocities along [100] direction...")
    sound_velocities = calculate_sound_velocities(C, density, np.array([1,0,0])) # Example direction [100]
    print("[SUCCESS] Sound velocities calculated.")

    # Defect and Stress Analysis (simplified)
    if auto_mode:
        pressure = ISOTROPIC_PRESSURE
        print(f"[INFO] Auto mode: Using pressure = {pressure} GPa")
    else:
        pressure = None
        while pressure is None:
            try:
                pressure_str = input("\n[INPUT] Enter isotropic pressure to apply in GPa (e.g., 10.0, or 0 for no stress): ")
                pressure = float(pressure_str)
            except ValueError:
                print("[ERROR] Invalid input. Please enter a number.")
    
    stressed_C = None
    if pressure > 0:
        print(f"[INFO] Applying {pressure} GPa isotropic pressure...")
        stressed_C = apply_isotropic_stress(C, pressure)
        print("[SUCCESS] Stress applied. Stressed matrix:\n", stressed_C)
    else:
        print("[INFO] No isotropic pressure applied.")

    # 2. Generate and save text report
    print("\n[INFO] Generating analysis report...")
    report_str = report(sym, stab, vrh, poly_moduli, poly_anisotropy, thermo_props, sound_velocities, stressed_C=stressed_C)
    
    # Find and report min/max E, G, nu, beta
    print("[INFO] Searching for extremal values of E, G, nu, and beta...")
    # For E, G, nu, beta, we need to sample directions. Use a dense grid.
    n_points_extremal = 181 # Number of points for theta and phi for extremal search
    phi_vals_extremal = np.linspace(0, 2 * np.pi, n_points_extremal)
    theta_vals_extremal = np.linspace(0, np.pi, n_points_extremal)

    max_E, min_E = -np.inf, np.inf
    max_G, min_G = -np.inf, np.inf
    max_nu, min_nu = -np.inf, np.inf
    max_beta, min_beta = -np.inf, np.inf
    dir_max_E, dir_min_E = None, None
    dir_max_G, dir_min_G = None, None
    dir_max_nu, dir_min_nu = None, None
    dir_max_beta, dir_min_beta = None, None

    for p in track(phi_vals_extremal, desc="搜索极值方向 (E/beta)"):
        for t in theta_vals_extremal:
            n = np.array([
                np.sin(t) * np.cos(p),
                np.sin(t) * np.sin(p),
                np.cos(t)
            ])
            
            # Young's Modulus
            current_E = youngs_modulus_directional(S, n)
            if current_E > max_E:
                max_E = current_E
                dir_max_E = n
            if current_E < min_E:
                min_E = current_E
                dir_min_E = n

            # Shear Modulus and Poisson's Ratio (extremal in plane perpendicular to n)
            # These functions are placeholders in elastic_tools.py and need proper implementation.
            # For now, we'll skip the extremal search for G and nu to avoid errors.
            # g_min_n, g_max_n, nu_min_n, nu_max_n = extremal_shear_and_poisson(S4, n)

            # Linear Compressibility
            current_beta = linear_compressibility_directional(S, n)
            if current_beta > max_beta:
                max_beta = current_beta
                dir_max_beta = n
            if current_beta < min_beta:
                min_beta = current_beta
                dir_min_beta = n

    report_str += "\n\n=== Extremal Elastic Properties ===\n"
    report_str += f"Young's Modulus (E):\n"
    report_str += f"  Max E = {max_E:.6f} GPa along [{dir_max_E[0]:.3f}, {dir_max_E[1]:.3f}, {dir_max_E[2]:.3f}]\n"
    report_str += f"  Min E = {min_E:.6f} GPa along [{dir_min_E[0]:.3f}, {dir_min_E[1]:.3f}, {dir_min_E[2]:.3f}]\n"
    # report_str += f"Shear Modulus (G):\n"
    # report_str += f"  Max G = {max_G:.6f} GPa (normal to [{dir_max_G[0]:.3f}, {dir_max_G[1]:.3f}, {dir_max_G[2]:.3f}])\n"
    # report_str += f"  Min G = {min_G:.6f} GPa (normal to [{dir_min_G[0]:.3f}, {dir_min_G[1]:.3f}, {dir_min_G[2]:.3f}])\n"
    # report_str += f"Poisson's Ratio (nu):\n"
    # report_str += f"  Max nu = {max_nu:.6f} (load along [{dir_max_nu[0]:.3f}, {dir_max_nu[1]:.3f}, {dir_max_nu[2]:.3f}])\n"
    # report_str += f"  Min nu = {min_nu:.6f} (load along [{dir_min_nu[0]:.3f}, {dir_min_nu[1]:.3f}, {dir_min_nu[2]:.3f}])\n"
    report_str += f"Linear Compressibility (beta):\n"
    report_str += f"  Max beta = {max_beta:.6f} GPa^-1 along [{dir_max_beta[0]:.3f}, {dir_max_beta[1]:.3f}, {dir_max_beta[2]:.3f}]\n"
    report_str += f"  Min beta = {min_beta:.6f} GPa^-1 along [{dir_min_beta[0]:.3f}, {dir_min_beta[1]:.3f}, {dir_min_beta[2]:.3f}]\n"

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report_str)
    print(f"[SUCCESS] Analysis report saved to: {report_path}")
    print(report_str)

    # 3. Generate and save plots and data
    print("\n[INFO] Generating plots and data files...")

    # Get user input for angle ranges and resolution
    if auto_mode:
        angle_range_2d = ANGLE_RANGE_2D
        phi_range_3d = PHI_RANGE_3D
        theta_range_3d = THETA_RANGE_3D
        n_phi_3d = N_PHI_3D
        n_theta_3d = N_THETA_3D
        print(f"[INFO] Auto mode: Using configured plot parameters")
        print(f"[INFO] 2D angle range: {angle_range_2d}")
        print(f"[INFO] 3D phi range: {phi_range_3d}, theta range: {theta_range_3d}")
        print(f"[INFO] 3D resolution: {n_phi_3d} x {n_theta_3d} points")
    else:
        print("\n[INPUT] Enter angle ranges for 2D polar plots (in radians). Default is 0 to 2*pi.")
        while True:
            try:
                angle_range_str = input("Enter 2D plot angle range (start, end) e.g., '0, 6.28' or press Enter for default: ")
                if not angle_range_str:
                    angle_range_2d = (0.0, 2.0 * np.pi)
                else:
                    start, end = map(float, angle_range_str.split(','))
                    angle_range_2d = (start, end)
                break
            except ValueError:
                print("[ERROR] Invalid input. Please enter two comma-separated numbers or press Enter.")

        print("\n[INPUT] Enter phi and theta ranges for 3D surface plots (in radians). Default is 0 to 2*pi for phi, 0 to pi for theta.")
        while True:
            try:
                phi_range_str = input("Enter 3D plot phi range (start, end) e.g., '0, 6.28' or press Enter for default: ")
                if not phi_range_str:
                    phi_range_3d = (0.0, 2.0 * np.pi)
                else:
                    start, end = map(float, phi_range_str.split(','))
                    phi_range_3d = (start, end)
                break
            except ValueError:
                print("[ERROR] Invalid input. Please enter two comma-separated numbers or press Enter.")

        while True:
            try:
                theta_range_str = input("Enter 3D plot theta range (start, end) e.g., '0, 3.14' or press Enter for default: ")
                if not theta_range_str:
                    theta_range_3d = (0.0, np.pi)
                else:
                    start, end = map(float, theta_range_str.split(','))
                    theta_range_3d = (start, end)
                break
            except ValueError:
                print("[ERROR] Invalid input. Please enter two comma-separated numbers or press Enter.")

        n_phi_3d = None
        n_theta_3d = None
        while n_phi_3d is None:
            try:
                n_phi_str = input("\n[INPUT] Enter number of points for 3D phi resolution (e.g., 121, or press Enter for default 121): ")
                if not n_phi_str:
                    n_phi_3d = 121
                else:
                    n_phi_3d = int(n_phi_str)
                if n_phi_3d <= 0:
                    raise ValueError
                break
            except ValueError:
                print("[ERROR] Invalid input. Please enter a positive integer or press Enter.")

        while n_theta_3d is None:
            try:
                n_theta_str = input("[INPUT] Enter number of points for 3D theta resolution (e.g., 121, or press Enter for default 121): ")
                if not n_theta_str:
                    n_theta_3d = 121
                else:
                    n_theta_3d = int(n_theta_str)
                if n_theta_3d <= 0:
                    raise ValueError
                break
            except ValueError:
                print("[ERROR] Invalid input. Please enter a positive integer or press Enter.")

    # 2D plots for E, G, nu, beta, B in standard planes
    print("\n[INFO] Generating 2D section plots...")
    
    # 用于存储所有2D图的极值信息
    all_2d_extremal_data = {}
    _props_2d = ["E", "L", "G", "nu", "B"]
    _bar2d = ProgressBar(len(_props_2d) * 3, desc="生成2D截面图")

    for prop_type in _props_2d:
        all_2d_extremal_data[prop_type] = {}
        
        for plane in ["xy", "yz", "xz"]:
            # 对于体弹性模量，使用刚度矩阵C
            matrix_to_use = C if prop_type == "B" else S
            
            png_path = os.path.join(output_dir, f"{prop_type}_2D_{plane}.png")
            result = plot_2d_polar(matrix_to_use, prop_type, plane, filename=png_path, angle_range=angle_range_2d)
            
            # 提取极值信息
            if prop_type in ['G', 'nu']:
                theta, max_values, min_values, extremal_info = result
            else:
                theta, values, extremal_info = result
            
            all_2d_extremal_data[prop_type][plane] = extremal_info

            # 导出 2D 曲线数据 (Origin 可直接读)
            dat2d = os.path.join(output_dir, f"{prop_type}_2D_{plane}_data.txt")
            if prop_type in ['G', 'nu']:
                _save_2d_polar_data(dat2d, prop_type, plane, theta, max_values, min_values)
            else:
                _save_2d_polar_data(dat2d, prop_type, plane, theta, values)
            
            # 为泊松比额外生成绝对值2D图
            if prop_type == "nu":
                png_path_abs = os.path.join(output_dir, f"{prop_type}_2D_{plane}_abs.png")
                result_abs = plot_2d_polar(S, prop_type, plane, filename=png_path_abs, angle_range=angle_range_2d, use_abs=True)
                theta_abs, max_values_abs, min_values_abs, extremal_info_abs = result_abs
                all_2d_extremal_data[f"{prop_type}_abs"] = all_2d_extremal_data.get(f"{prop_type}_abs", {})
                all_2d_extremal_data[f"{prop_type}_abs"][plane] = extremal_info_abs
                _save_2d_polar_data(os.path.join(output_dir, f"{prop_type}_2D_{plane}_abs_data.txt"),
                                    prop_type, plane, theta_abs, max_values_abs, min_values_abs)
            _bar2d.update(info=f"{prop_type}_{plane}")
    _bar2d.close()

    # 3D plots for E, G, nu, beta, B
    all_extremal_data = {}
    _props_3d = ["E", "L", "G", "nu", "B"]
    _bar3d = ProgressBar(len(_props_3d), desc="生成3D曲面+Origin数据")
    for prop_type in _props_3d:
        png_3d_path = os.path.join(output_dir, f"{prop_type}_3D_surface.png")
        if prop_type == "B": # Bulk modulus needs C matrix
            phi, theta, values, extremal_data = plot_3d_surface(C, prop_type, n_phi=n_phi_3d, n_theta=n_theta_3d, filename=png_3d_path, phi_range=phi_range_3d, theta_range=theta_range_3d)
        else:
            phi, theta, values, extremal_data = plot_3d_surface(S, prop_type, n_phi=n_phi_3d, n_theta=n_theta_3d, filename=png_3d_path, phi_range=phi_range_3d, theta_range=theta_range_3d)
        
        all_extremal_data[prop_type] = extremal_data

        # 为泊松比额外生成绝对值图
        if prop_type == "nu":
            png_3d_abs_path = os.path.join(output_dir, f"{prop_type}_3D_surface_abs.png")
            phi_abs, theta_abs, values_abs, extremal_data_abs = plot_3d_surface(S, prop_type, n_phi=n_phi_3d, n_theta=n_theta_3d, filename=png_3d_abs_path, phi_range=phi_range_3d, theta_range=theta_range_3d, use_abs=True)
            
            # Export absolute value data for Origin
            save_xyz_grid(os.path.join(output_dir, f"{prop_type}_3D_data_xyz_abs.txt"), phi_abs, theta_abs, values_abs)
            save_matrix(os.path.join(output_dir, f"{prop_type}_3D_data_matrix_abs.txt"), values_abs)
            extremal_dict_abs = {
                'max': (extremal_data_abs['max_value'], extremal_data_abs['max_direction']),
                'min': (extremal_data_abs['min_value'], extremal_data_abs['min_direction'])
            }
            save_extremal_values(os.path.join(output_dir, f"{prop_type}_extremal_values_abs.txt"), f"|{prop_type}|", extremal_dict_abs)

        # Export data for Origin
        save_xyz_grid(os.path.join(output_dir, f"{prop_type}_3D_data_xyz.txt"), phi, theta, values)
        save_matrix(os.path.join(output_dir, f"{prop_type}_3D_data_matrix.txt"), values)
        extremal_dict = {
            'max': (extremal_data['max_value'], extremal_data['max_direction']),
            'min': (extremal_data['min_value'], extremal_data['min_direction'])
        }
        save_extremal_values(os.path.join(output_dir, f"{prop_type}_extremal_values.txt"), prop_type, extremal_dict)
        _bar3d.update(info=f"{prop_type} (max={extremal_data['max_value']:.1f})")
    _bar3d.close()

        # Interactive 3D plot (placeholder)
        # html_path = os.path.join(output_dir, f"{prop_type}_3D_interactive.html")
        # plot_interactive_3d_surface(S, prop_type, html_path)
        # print(f"[SUCCESS] Saved interactive 3D plot: {html_path}")
    
    # 保存2D截面图的极值信息
    print(f"\n[INFO] Saving 2D section extremal values...")
    extremal_2d_path = os.path.join(output_dir, "2D_section_extremal_values.txt")
    with open(extremal_2d_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("2D Section Extremal Values and Corresponding Crystal Directions\n")
        f.write("=" * 80 + "\n\n")
        
        for prop_type in ["E", "L", "G", "nu", "nu_abs", "B"]:
            if prop_type not in all_2d_extremal_data:
                continue
                
            prop_name = "Poisson's Ratio (Absolute)" if prop_type == "nu_abs" else PROPERTY_NAMES.get(prop_type.replace("_abs", ""), prop_type)
            unit = PROPERTY_UNITS.get(prop_type.replace("_abs", ""), '')
            unit_str = f' {unit}' if unit else ''
            
            f.write(f"\n{'=' * 80}\n")
            f.write(f"{prop_name}\n")
            f.write(f"{'=' * 80}\n\n")
            
            for plane in ["xy", "yz", "xz"]:
                if plane not in all_2d_extremal_data[prop_type]:
                    continue
                    
                extremal_info = all_2d_extremal_data[prop_type][plane]
                
                f.write(f"{plane.upper()} Plane:\n")
                f.write(f"{'-' * 80}\n")
                f.write(f"  Maximum Value: {extremal_info['max_value']:.6f}{unit_str}\n")
                f.write(f"    Direction: [{extremal_info['max_direction'][0]:.6f}, "
                       f"{extremal_info['max_direction'][1]:.6f}, "
                       f"{extremal_info['max_direction'][2]:.6f}]\n")
                f.write(f"    Angle in plane: {extremal_info['max_angle_deg']:.2f}°\n")
                f.write(f"\n")
                f.write(f"  Minimum Value: {extremal_info['min_value']:.6f}{unit_str}\n")
                f.write(f"    Direction: [{extremal_info['min_direction'][0]:.6f}, "
                       f"{extremal_info['min_direction'][1]:.6f}, "
                       f"{extremal_info['min_direction'][2]:.6f}]\n")
                f.write(f"    Angle in plane: {extremal_info['min_angle_deg']:.2f}°\n")
                f.write(f"\n")
    
    print(f"[SUCCESS] Saved 2D section extremal values: {extremal_2d_path}")
    
    # 计算主轴方向的值
    print(f"\n[INFO] Calculating values along principal axes...")
    principal_axes = {
        '[100]': np.array([1.0, 0.0, 0.0]),
        '[010]': np.array([0.0, 1.0, 0.0]),
        '[001]': np.array([0.0, 0.0, 1.0])
    }
    
    principal_axis_values = {}
    from .elastic_tools import shear_modulus_directional, poisson_ratio_directional
    
    for prop_type in ["E", "L", "G", "nu", "B"]:
        principal_axis_values[prop_type] = {}
        for axis_name, direction in principal_axes.items():
            if prop_type == "E":
                value = youngs_modulus_directional(S, direction)
            elif prop_type == "L":
                value = linear_compressibility_directional(S, direction)
            elif prop_type == "G":
                # For shear modulus along principal axes, use return_extrema to get max/min
                value_max, value_min = shear_modulus_directional(S, direction, return_extrema=True)
                value = (value_max, value_min)
            elif prop_type == "nu":
                value_max, value_min = poisson_ratio_directional(S, direction, return_extrema=True)
                value = (value_max, value_min)
            elif prop_type == "B":
                value = bulk_modulus_directional(C, direction)
            
            principal_axis_values[prop_type][axis_name] = value
    
    # 保存所有3D极值的汇总文件(包含主轴方向的值)
    summary_extremal_path = os.path.join(output_dir, "all_extremal_values_summary.txt")
    print(f"[INFO] Saving summary of all extremal values (including principal axes) to {summary_extremal_path}...")
    with open(summary_extremal_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Extremal Values and Principal Axis Values\n")
        f.write("=" * 80 + "\n\n")
        
        for prop_type in ["E", "L", "G", "nu", "B"]:
            if prop_type not in all_extremal_data:
                continue
                
            prop_name = PROPERTY_NAMES.get(prop_type, prop_type)
            unit = PROPERTY_UNITS.get(prop_type, '')
            unit_str = f' {unit}' if unit else ''
            
            extremal_data = all_extremal_data[prop_type]
            max_dir = extremal_data['max_direction']
            min_dir = extremal_data['min_direction']
            
            f.write(f"\n{'=' * 80}\n")
            f.write(f"{prop_name}\n")
            f.write(f"{'=' * 80}\n\n")
            
            f.write(f"3D Extremal Values:\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"  Maximum Value: {extremal_data['max_value']:.6f}{unit_str}\n")
            f.write(f"    Direction: [{max_dir[0]:.6f}, {max_dir[1]:.6f}, {max_dir[2]:.6f}]\n")
            f.write(f"\n")
            f.write(f"  Minimum Value: {extremal_data['min_value']:.6f}{unit_str}\n")
            f.write(f"    Direction: [{min_dir[0]:.6f}, {min_dir[1]:.6f}, {min_dir[2]:.6f}]\n")
            f.write(f"\n")
            
            f.write(f"Principal Axis Values:\n")
            f.write(f"{'-' * 80}\n")
            for axis_name in ['[100]', '[010]', '[001]']:
                if prop_type in ["nu", "G"]:
                    # For Poisson's ratio and shear modulus, show both max and min
                    value_max, value_min = principal_axis_values[prop_type][axis_name]
                    f.write(f"  {axis_name}: Max = {value_max:.6f}{unit_str}, Min = {value_min:.6f}{unit_str}\n")
                else:
                    value = principal_axis_values[prop_type][axis_name]
                    f.write(f"  {axis_name}: {value:.6f}{unit_str}\n")
            f.write(f"\n")
    
    print(f"[SUCCESS] Saved extremal values summary with principal axes: {summary_extremal_path}")

    print("\n[INFO] All plots and data files generated.")

if __name__ == "__main__":
    # 完全自动化模式，无需任何用户输入
    # 所有参数在config.py中配置
    # 如果需要交互模式，将auto_mode改为False
    main(INPUT_MATRIX_FILE, OUTPUT_DIRECTORY, auto_mode=True)



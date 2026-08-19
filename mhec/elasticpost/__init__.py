"""
elasticpost: 弹性性质后处理子包。

提供弹性矩阵分析、方向性力学性质计算、VRH 平均、
稳定性判据、2D/3D 绘图等功能。
"""

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
    report,
    suggest_matrix_correction,
    swap_c44_c66,
    force_symmetry,
    calculate_thermodynamic_properties,
    calculate_sound_velocities,
    apply_isotropic_stress,
    calculate_polycrystalline_moduli,
    calculate_polycrystalline_anisotropy,
)
from .elastic_calculator import main as run_elastic_analysis
from .config import (
    DENSITY,
    MOLAR_MASS,
    ISOTROPIC_PRESSURE,
    INPUT_MATRIX_FILE,
    OUTPUT_DIRECTORY,
    AUTO_APPLY_CORRECTION,
    FORCE_SYMMETRY,
)

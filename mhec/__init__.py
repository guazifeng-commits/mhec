"""MHEC: MLFF-accelerated High-temperature Elastic Constants (VASP AIMD + MLFF)"""
__version__ = "1.0.0"
__author__ = "Dr. Feng Xing"
__email__ = "fengxing@amgm.ac.cn"

from .crystal_system import CrystalSystem, identify_crystal_system
from .config import MHECConfig
from .elastic_fitter import ElasticFitter
from .stress_extractor import StressExtractor
from .plugins import PluginRegistry, MechanicalPropertiesPlugin, AIMDPostProcessorPlugin
from .adapters import ElasticPostAdapter, AIMDPostAdapter
from .mlff_validator import (
    MLFFValidationResult, parse_ml_logfile,
    generate_validation_dirs, collect_parity_data,
    plot_parity, write_parity_report, ParityData,
)
from .thermal_expansion import compute_thermal_expansion, ThermalExpansionResult
from .adiabatic import (
    isothermal_to_adiabatic, run_conversion as run_adiabatic_conversion,
    AdiabaticResult,
)

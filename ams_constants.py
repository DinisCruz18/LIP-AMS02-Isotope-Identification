"""
====================================================================================
AMS-02 CONSTANTS & HYPERPARAMETERS
====================================================================================
Centralized module for all physical quantities, hardware calibrations, and 
analysis hyperparameters of the project.

Guarantees absolute consistency (Single Source of Truth) between the 
Monte Carlo (MC) generator and the analytical Template engine.
"""

# ===================================================================
# 1. ANALYSIS PARAMETERS (User Configurations)
# ===================================================================
# --- Monte Carlo Simulation ---
TARGET_EVENTS = 1000000         # Number of valid events to record in the TTree

# Probability of 0.5 (50/50) forces equal statistics for both 
# isotopes, which is ideal to reduce the error in template generation.
# In a real cosmic flux analysis, ~0.3 (30% 10B) would be used.
PROB_B10 = 0.3                 # Probability of injecting Boron-10
P_MIN = 1.0                    # Minimum generated rigidity [GV]
P_MAX = 1000.0                 # Maximum generated rigidity [GV]

# --- Fitting Engine and Templates ---
# Centralizing the binning ensures that the SciPy arrays and ROOT histograms 
# have strictly the same axes, preventing crashes in TFractionFitter.
N_BINS = 200                   # Geometric resolution of the mass analysis
M_MIN = 5.0                    # Lower limit of the mass axis [GeV/c^2]
M_MAX = 16.0                   # Upper limit of the mass axis [GeV/c^2]
SIMULATION_FILENAME = "ams_b10_b11_simulation.root"

# --- Analysis Channels ---
# Defines the detectors and velocity (beta) windows to be evaluated.
# Centralizing this ensures the template generator and the fraction fitter 
# iterate exactly over the same physical phase space.
ANALYSIS_CHANNELS = [
    {'detector': 'TOF', 'beta_bin': (0.80, 0.85)},
    {'detector': 'NaF', 'beta_bin': (0.90, 0.95)},
    {'detector': 'AGL', 'beta_bin': (0.96, 0.99)}  
]

# Global flag to enable/disable Tracker's convoluted integration
USE_TRACKER_RESOLUTION = True  

# ===================================================================
# 2. FUNDAMENTAL PHYSICAL CONSTANTS (Immutable)
# ===================================================================
Z_CHARGE = 5                   # Atomic number of Boron (Z)
MASS_NUCLEON = 0.9315          # Reference mass of the nucleon [GeV/c^2]
MASS_B10 = 10.0129             # Exact rest mass of 10B [GeV/c^2]
MASS_B11 = 11.0093             # Exact rest mass of 11B [GeV/c^2]
C_SPEED = 29.979               # Speed of light [cm/ns]

# ===================================================================
# 3. AMS-02 DETECTOR GEOMETRY (Local reference frame)
# ===================================================================
# --- TOF (Time-of-Flight) ---
TOF_SIDE = 130.0               # Side of the square scintillator plane [cm]
TOF_HALF_SIDE = TOF_SIDE / 2.0 # Half-side for acceptance calculation [cm]
TOF_Z_DIST = 127.3             # Absolute Z distance between TOF planes [cm]

# --- RICH (Ring Imaging Cherenkov) ---
RICH_NAF_SIDE = 34.5           # Side of the central NaF square [cm]
RICH_AGL_RADIUS = 57.0         # Maximum radius of the outer Aerogel ring [cm]
RICH_Z_DIST = 135.3            # Z distance between upper TOF and radiator [cm]

# ===================================================================
# 4. INSTRUMENTAL RESOLUTION (Calibration Parameters)
# ===================================================================
# --- Tracker ---
# Relative uncertainty of the rigidity measurement (sigma_P / P)
TRACKER_SIGMA_REL = 0.04       # percentage constant error

# --- TOF (Time-of-Flight) ---
# Empirical parameterization: sigma_t = sqrt((A/Z)^2 + B^2)
TOF_A_PS = 159.0               # Charge-dependent statistical term [ps]
TOF_B_PS = 79.0                # Asymptotic limit term [ps]

# --- RICH (Sodium Fluoride - NaF) ---
# Parameterization: sigma_beta / beta = A - B * exp(1 - E_kn / E0)
RICH_NAF_N = 1.3340            # Refractive index
RICH_NAF_THR = 1.0 / 1.3340    # Cherenkov kinematic threshold (beta > 1/n)
RICH_NAF_A = 12.73e-4          # Resolution parameter A
RICH_NAF_B = 3.76e-4           # Resolution parameter B
RICH_NAF_E0 = 0.938            # Stabilization kinetic energy [GeV/n]

# --- RICH (Aerogel - AGL) ---
RICH_AGL_N = 1.050             # Refractive index
RICH_AGL_THR = 1.0 / 1.050     # Cherenkov kinematic threshold (beta > 1/n)
RICH_AGL_A = 3.57e-4           # Resolution parameter A
RICH_AGL_B = 0.61e-4           # Resolution parameter B
RICH_AGL_E0 = 2.983            # Stabilization kinetic energy [GeV/n]

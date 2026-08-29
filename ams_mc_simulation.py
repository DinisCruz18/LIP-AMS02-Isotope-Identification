import numpy as np
import pandas as pd
from ams_constants import *

class AMS02Simulator:
    """
    Monte Carlo Simulator of the AMS-02 Detector for Cosmic Rays.
    
    This class simulates the injection and detection of Boron isotopes (10B and 11B) in the 
    AMS-02 magnetic spectrometer installed on the International Space Station (ISS). It propagates the relativistic 
    kinematics from the top of the detector and applies the resolution functions (smearing) of the 
    Tracker, TOF, and RICH subsystems. The output simulates the data required for the 
    construction of analytical mass templates.

    ====================================================================================
    CURRENT CONSIDERATIONS (Simplifications compared to the Script)
    ====================================================================================
    1. A rectilinear propagation of the particle is assumed for the purpose 
       of geometric acceptance (TOF Trigger and impact on the RICH plane). The actual 
       trajectory curvature induced by the 0.15 T magnetic field is not traced step-by-step.
       
    2. Tracker Resolution: The relative uncertainty of the rigidity measurement (sigma_P / P) 
       is fixed at a constant value (4%).
       
    3. Cosmic Spectrum: The generation follows a log-uniform distribution (P^-1) 
       reweighted to a generic power-law primary flux (P^-2.7).
       
    4. Geomagnetic Acceptance: It is assumed that all generated particles overcome the Earth's 
       magnetic field barrier (Cutoff Rigidity, P_cut), neglecting the orbital effect 
       of the ISS on geomagnetic latitudes.

    ====================================================================================
    FUTURE WORK 
    ====================================================================================
    - Tracker: Implement the resolution dependent on the Maximum Detectable Rigidity (MDR), 
      reflecting the error in the sagitta measurement at high energies (sigma_P / P = P / MDR).
      
    - Replace the generic power law with the official SBPL (Smoothly Broken Power Law) parameterization 
      of the Local Interstellar Spectrum (LIS) flux.
      
    - Introduce the Fisk Solar Modulation (Force-Field Approximation) for 
      rigidities < 10 GV.
      
    - Implement a dynamic filtering based on the P_cut calculated through 
      the IGRF geomagnetic models along the ISS orbit.
    """
    
    def __init__(self):
        """
        Initializes the Monte Carlo simulator with the physical and geometric parameters of the AMS-02.
        """
        # ===================================================================
        # 1. FUNDAMENTAL PHYSICAL PARAMETERS
        # ===================================================================
        self.p_min = P_MIN          # Minimum rigidity [GV]
        self.p_max = P_MAX          # Maximum rigidity [GV]
        self.z_charge = Z_CHARGE    # Isotope charge (Z=5 for Boron)
        self.prob_b10 = PROB_B10    # Probability of 10B isotope generation
        
        # Rest masses [GeV/c^2]
        self.mass_b10 = MASS_B10 
        self.mass_b11 = MASS_B11 
        self.mass_nucleon = MASS_NUCLEON  # Reference mass of the nucleon [GeV/c^2]
        self.c_speed = C_SPEED       # Speed of light [cm/ns]

        # ===================================================================
        # 2. DETECTOR GEOMETRY (Local reference frame in cm)
        # ===================================================================
        # TOF (Time of Flight) - Upper and lower planes
        self.tof_side = TOF_SIDE                   # Side of the square plane [cm]
        self.tof_half_side = TOF_HALF_SIDE         # Half-side for acceptance limits [cm]
        self.tof_z_dist = TOF_Z_DIST               # Absolute Z distance between upper and lower TOF [cm]
        
        # RICH (Ring Imaging Cherenkov)
        self.rich_naf_side = RICH_NAF_SIDE               # Side of the central NaF square [cm]
        self.rich_agl_radius = RICH_AGL_RADIUS           # Maximum radius of the outer Aerogel ring [cm]
        self.rich_z_dist = RICH_Z_DIST                   # Absolute Z distance between upper TOF and RICH radiator [cm]

        # ===================================================================
        # 3. INSTRUMENTAL RESOLUTION AND UNCERTAINTY
        # ===================================================================
        # Tracker: Constant relative uncertainty
        self.tracker_sigma_rel = TRACKER_SIGMA_REL # sigma_P / P = 4%

        # Flag to toggle Tracker's ideal vs realistic resolution
        self.use_tracker_resolution = USE_TRACKER_RESOLUTION
        
        # TOF: Parameterization of the temporal resolution
        # Formula: sigma_t = sqrt((A/Z)^2 + B^2)
        self.tof_A_ps = TOF_A_PS  # Charge-dependent statistical term [ps]
        self.tof_B_ps = TOF_B_PS  # Intrinsic electronics/scintillators term [ps]

        # RICH: Parameterization of the relative velocity resolution
        # Formula: sigma_beta / beta = A - B * exp(1 - E_kn / E0)
        
        # Sodium Fluoride (NaF) radiator for Z=5
        self.rich_naf_n = RICH_NAF_N                     # Refractive index
        self.rich_naf_thr = RICH_NAF_THR                 # Kinematic threshold (beta > 1/n)
        self.rich_naf_A = RICH_NAF_A
        self.rich_naf_B = RICH_NAF_B
        self.rich_naf_E0 = RICH_NAF_E0                   # Reference energy [GeV/n]

        # Aerogel (AGL) radiator for Z=5
        self.rich_agl_n = RICH_AGL_N                     # Refractive index
        self.rich_agl_thr = RICH_AGL_THR                 # Kinematic threshold (beta > 1/n)
        self.rich_agl_A = RICH_AGL_A
        self.rich_agl_B = RICH_AGL_B
        self.rich_agl_E0 = RICH_AGL_E0                   # Reference energy [GeV/n]

        # ===================================================================
        # 4. PRE-CALCULATED VARIABLES
        # ===================================================================
        # Logarithmic mean of the rigidity for the statistical weight normalization (w)
        # Formula: P_mean = (P_max - P_min) / ln(P_max / P_min)
        self.p_mean = (self.p_max - self.p_min) / np.log(self.p_max / self.p_min)

    def generate_rigidity(self) -> float:
        """
        Generates the true rigidity (P0) [GV] according to a flat distribution in log(P).
        
        Uses the CDF (Cumulative Distribution Function) inversion method.
        Inversion formula: P0 = P_min * (P_max / P_min)^u, where u belongs to [0, 1].
        
        Returns:
            float: Generated rigidity (P0) in GV.
        """
        u = np.random.uniform(0.0, 1.0)
        return self.p_min * (self.p_max / self.p_min)**u
    
    def mock_cosmic_flux(self, p: float) -> float:
        """
        Calculates the cosmic flux J(P0) for the given rigidity.
        
        CURRENT: Generic Power Law approximation J(P) ~ P^-2.7.
        
        FUTURE WORK: 
        Implement the official SBPL (Smoothly Broken Power Law) parameterization of the AMS-02:
        J_LIS(P) = C * (P/P0)^(-alpha) * [1 + (P/Ps)^(delta_alpha/s)]^s
        and add the Fisk Solar Modulation (Force-Field Approximation) for P < 10 GV.
        
        Args:
            p (float): True rigidity (P0) in GV.
            
        Returns:
            float: Value proportional to the expected flux.
        """
        return p ** -2.7

    def calculate_weight(self, p0: float) -> float:
        """
        Calculates the statistical weight (w) of the event to correct the log-uniform generation,
        forcing the final distribution to mirror the real cosmic flux J(P0).
        
        Formula:
        w(P0) = (P0 * J(P0)) / (<P0> * J(<P0>))
        
        Args:
            p0 (float): True rigidity (P0) in GV.
            
        Returns:
            float: Dimensionless statistical weight (w).
        """
        numerator = p0 * self.mock_cosmic_flux(p0)
        denominator = self.p_mean * self.mock_cosmic_flux(self.p_mean)
        return numerator / denominator

    def generate_kinematics(self) -> tuple:
        """
        Generates the spatial (x, y) and angular (cos_theta, phi) coordinates of the 
        incident particle assuming an isotropic flux at the top of the detector.
        
        The spatial distribution is uniform over the upper LTOF area:
        x, y in [-tof_half_side, tof_half_side]
        
        The isotropic angular distribution requires flat distributions in cos(theta) and phi:
        d^2N / (d(cos_theta) d(phi)) = 1
        
        Returns:
            tuple: (x [cm], y [cm], cos_theta [dimensionless], phi [rad])
        """
        u_x = np.random.uniform(0, 1)
        u_y = np.random.uniform(0, 1)
        u_theta = np.random.uniform(0, 1)
        u_phi = np.random.uniform(0, 1)

        # Position on the upper TOF plane (z = 0)
        x = -self.tof_half_side + u_x * self.tof_side
        y = -self.tof_half_side + u_y * self.tof_side

        # Incidence angles (limited to the downward hemisphere)
        # Being isotropic, cos_theta maps linearly from 0 to 1.
        cos_theta = u_theta
        phi = u_phi * 2 * np.pi
        
        return x, y, cos_theta, phi

    def geometric_trigger_passed(self, x: float, y: float, cos_theta: float, phi: float) -> bool:
        """
        Simulates the AMS-02 Trigger condition (Geometric Acceptance).
        Requires the particle to cross the active dimensions of the lower TOF.
        
        Linear propagation equations (assuming negligible magnetic deflection in the trigger):
        delta_x = delta_z * tan(theta) * cos(phi)
        delta_y = delta_z * tan(theta) * sin(phi)
        
        Args:
            x (float): X position on the upper LTOF [cm].
            y (float): Y position on the upper LTOF [cm].
            cos_theta (float): Cosine of the zenith incidence angle.
            phi (float): Azimuthal angle [rad].
            
        Returns:
            bool: True if it crosses the lower LTOF, False otherwise.
        """
        # Avoid division by zero for perfectly horizontal particles
        if cos_theta == 0.0:
            return False
            
        # Optimized trigonometric extraction
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        tan_theta = sin_theta / cos_theta
        
        # Propagation to the lower plane z = -tof_z_dist
        # The movement is in -Z, hence the negative translation.
        x_lower = x + (-self.tof_z_dist) * tan_theta * np.cos(phi)
        y_lower = y + (-self.tof_z_dist) * tan_theta * np.sin(phi)

        # Check if the impact point is inside the lower TOF square
        if abs(x_lower) <= self.tof_half_side and abs(y_lower) <= self.tof_half_side:
            return True
            
        return False

    def get_rich_radiator(self, x_top: float, y_top: float, cos_theta: float, phi: float) -> float:
        """
        Linearly extrapolates the particle trajectory to the radiator plane 
        of the RICH detector and geometrically determines the hit region.
        
        Dual Radiator Geometry:
        - NaF (Center): 34.5 x 34.5 cm square.
        - AGL (Outer): Circular ring with a maximum radius of 57.0 cm.
        
        Args:
            x_top (float): Generated X position at the top of the detector [cm].
            y_top (float): Generated Y position at the top of the detector [cm].
            cos_theta (float): Cosine of the zenith angle.
            phi (float): Azimuthal angle [rad].
            
        Returns:
            float: 1 (NaF), 2 (AGL), or -999.0 (if it misses both radiators).
        """
        # For perfectly horizontal trajectories
        if cos_theta == 0.0:
            return -999.0
            
        # Trigonometric extraction
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        tan_theta = sin_theta / cos_theta
        
        # Linear propagation to the RICH Z plane
        x_rich = x_top + (-self.rich_z_dist) * tan_theta * np.cos(phi)
        y_rich = y_top + (-self.rich_z_dist) * tan_theta * np.sin(phi)
        
        # NaF square (34.5 x 34.5 cm)
        naf_half_side = self.rich_naf_side / 2.0
        if abs(x_rich) <= naf_half_side and abs(y_rich) <= naf_half_side:
            return 1
            
        # Aerogel circle (Radius 57 cm)
        # Using r^2 <= R^2 avoids the heavy np.sqrt() function
        r_rich_squared = x_rich**2 + y_rich**2
        if r_rich_squared <= self.rich_agl_radius**2:
            return 2
            
        # Failed RICH acceptance
        return -999.0
    
    def smear_velocity_tof(self, beta_0: float, cos_theta: float) -> float:  
        """
        Simulates the velocity (beta) measurement by the Time-of-Flight (TOF) system,
        applying the intrinsic statistical uncertainty of the detector's temporal resolution.
        
        The velocity is reconstructed by measuring the time of flight (Delta t) 
        required to cross the distance (d) between the upper and lower planes:
        1 / beta = c * (Delta t / d)
        
        The temporal resolution (sigma_t) is modeled empirically and depends on the charge (Z):
        sigma_t = sqrt((A/Z)^2 + B^2)
        
        Args:
            beta_0 (float): True velocity of the particle (v/c).
            cos_theta (float): Cosine of the zenith angle (to calculate the actual path).
            
        Returns:
            float: Velocity measured by the TOF (beta_recon), physically constrained to < 1.0.
        """
        # Temporal resolution sigma_t (conversion from picoseconds to nanoseconds)
        sigma_t_ns = np.sqrt((self.tof_A_ps / self.z_charge)**2 + self.tof_B_ps**2) / 1000.0
        
        # Calculation of the actual distance traveled d [cm]
        # Where d is the hypotenuse of the trajectory between the two planes: d = delta_z / cos(theta)
        d_cm = self.tof_z_dist / cos_theta
        
        # True time of flight (t_true) [ns]
        # Derived from the definition of velocity: beta * c = d / t => t = d / (beta * c)
        t_true = d_cm / (beta_0 * self.c_speed)
        
        # Simulation of the temporal measurement (Smearing)
        # The measured time fluctuates around the true time according to a Normal distribution
        t_recon = np.random.normal(loc=t_true, scale=sigma_t_ns)
        
        # Extreme fluctuations cannot generate negative or zero flight times
        if t_recon <= 0.0:
            return 0.000001
            
        # Velocity reconstruction from the imperfect measured time
        beta_recon = d_cm / (t_recon * self.c_speed)
        
        # The particle cannot be measured moving faster than light
        return min(beta_recon, 0.999999)

    def smear_velocity_rich(self, beta_0: float, e_kn: float, radiator: str) -> float:
        """
        Simulates the velocity (beta) measurement by the RICH (Ring Imaging Cherenkov) detector,
        applying the empirical resolution of the hit radiator (NaF or AGL).
        
        Cherenkov light emission only occurs if the particle exceeds the speed of light 
        in the dielectric medium (Kinematic Threshold):
        beta_thr = 1 / n
        
        The relative velocity resolution (sigma_beta / beta) is derived from the error in the 
        reconstruction of the Cherenkov angle and parameterized by the kinetic energy per nucleon (E_kn):
        sigma_beta / beta = A - B * exp(1 - E_kn / E0)
        
        Args:
            beta_0 (float): True velocity of the particle (v/c).
            e_kn (float): Kinetic energy per nucleon [GeV/n].
            radiator (str): Material of the hit radiator ('NaF', 'AGL', or None).
            
        Returns:
            float or None: Velocity measured by the RICH. Returns None if the particle 
                           does not produce light (beta_0 <= threshold) or misses the detector.
        """
        # Parameter selection based on the radiator
        if radiator == 'NaF':
            # Cherenkov threshold verification
            if beta_0 <= self.rich_naf_thr:
                return None
            sigma_beta_rel = self.rich_naf_A - self.rich_naf_B * np.exp(1.0 - (e_kn / self.rich_naf_E0))
            
        elif radiator == 'AGL':
            if beta_0 <= self.rich_agl_thr:
                return None
            sigma_beta_rel = self.rich_agl_A - self.rich_agl_B * np.exp(1.0 - (e_kn / self.rich_agl_E0))
            
        else:
            # Acceptance failure or invalid argument
            return None 
            
        # Calculation of the absolute standard deviation of the measurement
        sigma_beta = sigma_beta_rel * beta_0
        
        # Simulation of the measurement (Smearing)
        # Assuming that the angular error translates into a normal error in velocity
        beta_recon = np.random.normal(loc=beta_0, scale=sigma_beta)
        
        # beta must be strictly positive and cannot exceed c
        return min(max(beta_recon, 0.000001), 0.999999)

    def smear_rigidity_tracker(self, p0: float) -> float:
        """
        Simulates the rigidity (P) measurement by the Silicon Tracker, applying 
        the statistical uncertainty arising from the detector's spatial resolution.
        
        If the global flag 'USE_TRACKER_RESOLUTION' is disabled, the function 
        bypasses the smearing and returns an ideal measurement (P = P0). This allows 
        for controlled studies to isolate the intrinsic separation power of the TOF and RICH.
        
        CURRENT: Assumes a constant relative uncertainty (sigma_P / P = 4%).
        Formula: sigma_P = (sigma_P / P) * P0
        
        POSSIBLE FUTURE WORK:
        Implement the true resolution dependence on energy. 
        In a magnetic spectrometer, the error dominates at high energies because 
        the trajectory's sagitta decreases, approaching a straight line:
        sigma_P / P = P / MDR (Maximum Detectable Rigidity).
        
        Args:
            p0 (float): True rigidity of the particle [GV].
            
        Returns:
            float: Reconstructed rigidity (P) [GV], protected against non-physical values.
        """
        # Bypass smearing if the Tracker is assumed to be ideal (Controlled Study Mode)
        if not self.use_tracker_resolution:
            return p0

        # Calculation of the absolute Gaussian error
        sigma_p = self.tracker_sigma_rel * p0
        
        # Simulation of the measurement (Smearing)
        p_recon = np.random.normal(loc=p0, scale=sigma_p)
        
        # Prevent extreme fluctuations from generating negative rigidity.
        # Set a lower limit of 0.1 GV.
        return max(p_recon, 0.1)

    def process_kinematics(self, p0: float, event_mass: float) -> tuple:
        """
        Calculates the exact relativistic kinematics of the particle prior to any 
        instrumental interaction, using natural units (c = 1).
        
        Mathematical Relations:
        1. Linear Momentum: p = Z * P0
        2. Velocity: beta = p / np.sqrt(p^2 + m^2)
        3. Lorentz Factor: gamma = 1 / np.sqrt(1 - beta^2)
        4. Total Kinetic Energy: T0 = (gamma - 1) * m
        5. Kinetic Energy per Nucleon: E_kn = (gamma - 1) * m_nucleon
        
        Args:
            p0 (float): True rigidity (P0) in GV.
            event_mass (float): Exact mass of the isotope (10B or 11B) in GeV/c^2.
            
        Returns:
            tuple: (beta_0, gamma, T0 [GeV], E_kn [GeV/n])
        """
        # 1. Rigidity to Linear Momentum Conversion (p = Z * P)
        p_mom = self.z_charge * p0
        
        # 2. True Velocity (beta_0) from relativistic invariance
        beta_0 = p_mom / np.sqrt(p_mom**2 + event_mass**2)
        
        # 3. Lorentz Factor (gamma)
        gamma = 1.0 / np.sqrt(1.0 - beta_0**2)
        
        # 4. Kinetic Energies
        t_0 = (gamma - 1.0) * event_mass
        
        # Use the theoretical nucleon mass for consistency in the scale (GeV/n)
        e_kn = (gamma - 1.0) * self.mass_nucleon
        
        return beta_0, gamma, t_0, e_kn

    def simulate_event(self) -> dict:
        """
        Orchestrates the complete simulation of a single Monte Carlo event.
        
        Event Pipeline:
        1. Isotopic Selection: 10B or 11B based on the defined probability.
        2. Spatial Generation and Trigger: Evaluates the geometric acceptance of the TOF.
        3. True Kinematics: Generates the rigidity and calculates the truth.
        4. Detector Response: Applies the uncertainties (smearing) of the Tracker, TOF, and RICH.
        5. Reconstruction: Inverts the relativistic equations using only measured variables 
           to calculate the Kinetic Energy and Reconstructed Mass.
           Mass Formula: m = (Z * P_measured / beta_measured) * np.sqrt(1 - beta_measured^2)
           
        Returns:
            dict or None: Dictionary with all variables (generated and measured) 
                          ready for the ROOT Tree. Returns None if it fails the Trigger.
        """
        # ===================================================================
        # 1. ISOTOPIC SELECTION
        # ===================================================================
        if np.random.random() < self.prob_b10:
            event_mass = self.mass_b10
        else:
            event_mass = self.mass_b11

        # ===================================================================
        # 2. SPATIAL GENERATION AND TRIGGER
        # ===================================================================
        x, y, cos_theta, phi = self.generate_kinematics()
        
        # If the particle does not completely cross the detector, the event is discarded
        if not self.geometric_trigger_passed(x, y, cos_theta, phi):
            return None
            
        # ===================================================================
        # 3. TRUE KINEMATICS (Monte Carlo Truth)
        # ===================================================================
        p0 = self.generate_rigidity()
        w = self.calculate_weight(p0)
        
        beta_0, gamma, t_0, e_kn = self.process_kinematics(p0, event_mass)

        # ===================================================================
        # 4. DETECTOR RESPONSE (Smearing)
        # ===================================================================
        # Tracker: Rigidity Measurement
        p_recon = self.smear_rigidity_tracker(p0)
        
        # TOF: Velocity Measurement
        beta_tof = self.smear_velocity_tof(beta_0, cos_theta)
        
        # RICH: Radiator Identification and Velocity Measurement
        radiator = self.get_rich_radiator(x, y, cos_theta, phi)
        if radiator != -999.0:
            radiator_str = 'NaF' if radiator == 1 else 'AGL'
            beta_rich = self.smear_velocity_rich(beta_0, e_kn, radiator_str)
            if beta_rich is None:
                beta_rich = -999.0
        else:
            beta_rich = -999.0

        # ===================================================================
        # 5. VARIABLE RECONSTRUCTION (Data Analysis)
        # ===================================================================
        # Theoretical mass approximation (A * nucleon_mass) used for energy
        m_aprox = round(event_mass) * self.mass_nucleon        
        
        # 5.1 Kinetic Energy Reconstruction (T = (gamma - 1) * m_approx)
        t_measured_tof = (1.0 / np.sqrt(1.0 - beta_tof**2) - 1.0) * m_aprox if (beta_tof != -999.0 and beta_tof < 1.0) else -999.0
        t_measured_rich = (1.0 / np.sqrt(1.0 - beta_rich**2) - 1.0) * m_aprox if (beta_rich != -999.0 and beta_rich < 1.0) else -999.0

        # 5.2 Mass Reconstruction
        # Relativistic Inversion: m = (Z * P / beta) * np.sqrt(1 - beta^2)
        m_true = event_mass        

        m_tof = (self.z_charge * p_recon / beta_tof) * np.sqrt(1.0 - beta_tof**2) if (beta_tof != -999.0 and 0.0 < beta_tof < 1.0) else -999.0
        m_rich = (self.z_charge * p_recon / beta_rich) * np.sqrt(1.0 - beta_rich**2) if (beta_rich != -999.0 and 0.0 < beta_rich < 1.0) else -999.0

        # ===================================================================
        # DATA EXPORT (Dictionary for ROOT)
        # ===================================================================
        return {
            "Z": self.z_charge,
            "A": round(event_mass),
            "w": w,
            "x": x,
            "y": y,
            "cos_theta": cos_theta,
            "phi": phi,
            
            # True Quantities
            "P0": p0,
            "e_kn": e_kn,
            "T0": t_0,
            "beta_0": beta_0,
            "m_true": m_true,
            
            # Measured and Reconstructed Quantities
            "P": p_recon,                  
            "beta_TOF": beta_tof,
            "radiator": radiator,  
            "beta_RICH": beta_rich,
            "T_measured_TOF": t_measured_tof,   
            "T_measured_RICH": t_measured_rich,  
            "m_TOF": m_tof,                
            "m_RICH": m_rich               
        }
    
if __name__ == "__main__":
    sim = AMS02Simulator()
    
    n_events_to_generate = TARGET_EVENTS
    valid_events = []
    
    while len(valid_events) < n_events_to_generate:
        event = sim.simulate_event()
        if event is not None:
            valid_events.append(event)
            
    print(f"Simulated {n_events_to_generate} valid events (that passed the TOF trigger).")
    
    print("\nFirst 10 events")
    df_events = pd.DataFrame(valid_events)
    print(df_events.head(10).to_string(index=False))

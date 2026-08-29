import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
import concurrent.futures       # For CPU multiprocessing
from functools import partial   # To facilitate argument injection in parallel functions
from ams_constants import *

class AMSMassTemplateAnalytic:
    """
    Mathematical and Analytical Engine for Mass Template Generation for the AMS-02.
    
    This class implements the statistical formulation based on Bayes' Theorem to 
    derive theoretical probability distributions dN/dm (Analytical Templates). 
    These templates serve as a theoretical model for the precise separation and 
    identification of neighboring isotopes (Boron-10 and Boron-11) from the 
    simulated data.

    ====================================================================================
    PHYSICAL AND MATHEMATICAL FOUNDATIONS
    ====================================================================================
    1. Unlike Monte Carlo simulations 
       (which generate discrete events with statistical noise), this script numerically 
       solves continuous integrations of Probability Density Functions (PDFs).
       
    2. Instrumental Error Modeling (g_beta and g_P):
       - TOF: Evaluates the statistical uncertainty in the temporal domain t and applies 
         the Jacobian of the transformation to the velocity domain (|dt/dbeta|).
       - RICH: Models the resolution dependent on the emission threshold and kinetic 
         energy per nucleon (E_kn) for the NaF and Aerogel (AGL) radiators.
       - Tracker: Incorporates the convolution with the rigidity uncertainty (sigma_P).
       
    3. Change of Variables (Jacobian):
       The analytical transition from the measured phase space (P, beta) to the mass space (m) 
       requires multiplication by the determinant of the Jacobian |J| = (gamma * beta) / Z.

    ====================================================================================
    COMPUTATIONAL OPTIMIZATION
    ====================================================================================
    - The calculation of numerical integrations (SciPy's quad and dblquad) 
      is distributed in parallel across all logical threads of the processor via 
      `ProcessPoolExecutor`, reducing the processing time of the mass arrays.
    - Guarantees probability conservation through 
      trapezoidal integration, forcing the total area under each template to be exactly 1.
    """
    
    def __init__(self):
        """
        Initializes the analytical engine with the physical and resolution parameters of the AMS-02.
        
        These parameters exactly mirror those used in the Monte Carlo simulation 
        to ensure that the theoretical model (Template) perfectly fits 
        the generated data, validating Bayes' Theorem applied to the separation.
        """
        # ===================================================================
        # 1. FUNDAMENTAL PHYSICAL PARAMETERS
        # ===================================================================
        self.z_charge = Z_CHARGE       # Isotope charge (Z=5 for Boron)
        self.mass_nucleon = MASS_NUCLEON # Reference mass of the nucleon [GeV/c^2]
        self.c_speed = C_SPEED          # Speed of light [cm/ns]

        # ===================================================================
        # 2. TOF (Time-of-Flight) RESOLUTION
        # ===================================================================
        # Standard flight distance assumed for the analytical calculation
        self.tof_z_dist = TOF_Z_DIST        # Absolute Z distance between planes [cm]
        
        # Parameterization of the empirical temporal resolution of the AMS-02:
        # sigma_t = sqrt((A/Z)^2 + B^2)
        self.tof_A_ps = TOF_A_PS          # Charge-dependent statistical term [ps]
        self.tof_B_ps = TOF_B_PS           # Asymptotic limit term of the electronics [ps]

        # ===================================================================
        # 3. RICH (Ring Imaging Cherenkov) RESOLUTION
        # ===================================================================
        # Parameterization of the relative velocity resolution:
        # sigma_beta / beta = A - B * exp(1 - E_kn / E0)
        
        # Sodium Fluoride (NaF) radiator for Z=5
        self.rich_naf_n = RICH_NAF_N                  # Refractive index
        self.rich_naf_thr = RICH_NAF_THR # Cherenkov threshold (beta > 1/n)
        self.rich_naf_A = RICH_NAF_A               # Resolution parameter A
        self.rich_naf_B = RICH_NAF_B               # Resolution parameter B
        self.rich_naf_E0 = RICH_NAF_E0                  # Stabilization energy [GeV/n]
        
        # Aerogel (AGL) radiator for Z=5
        self.rich_agl_n = RICH_AGL_N                   # Refractive index
        self.rich_agl_thr = RICH_AGL_THR # Cherenkov threshold (beta > 1/n)
        self.rich_agl_A = RICH_AGL_A                 # Resolution parameter A
        self.rich_agl_B = RICH_AGL_B                 # Resolution parameter B
        self.rich_agl_E0 = RICH_AGL_E0                  # Stabilization energy [GeV/n]

        # ===================================================================
        # 4. TRACKER RESOLUTION
        # ===================================================================
        # CURRENT: Constant relative uncertainty of 4% (sigma_P / P = 0.04)
        # FUTURE WORK: Implement the Maximum Detectable Rigidity (MDR) curve to 
        # accommodate the error increase at high energies.
        self.tracker_sigma_rel = TRACKER_SIGMA_REL

    def cosmic_flux(self, p0: float) -> float:
        """
        Calculates the primary differential cosmic flux J(P0) [particles / (m^2 sr s GV)].
        
        In the analytical formulation (Bayes' Theorem), this term acts as the physical "Prior" 
        that shapes the theoretical rigidity distribution.
        
        CURRENT: Generic Power Law approximation for high energies.
        
        FUTURE WORK: 
        Implement the LIS (Local Interstellar Spectrum) via SBPL
        and the corresponding Fisk Solar Modulation for the top of the atmosphere (J_TOA).
        
        Args:
            p0 (float): True rigidity (P0) in GV.
            
        Returns:
            float: Value proportional to the expected flux.
        """
        # Safeguard to avoid non-physical domains in the power
        if p0 <= 0.0:
            return 0.0
            
        return p0 ** -2.7

    def acceptance_and_exposure(self, p0: float) -> float:
        """
        Calculates the product of the Geometric Acceptance G(P0) and the Relative Exposure 
        Time T(P0)/T_max due to the geomagnetic field shielding.
        
        CURRENT: Approximation to a constant (1.0). 
         
        For isotopic separation analyses (e.g., 10B and 11B) within the same 
        narrow bin, the variation in the efficiency of the AMS-02 and the geomagnetic 
        latitude (Cutoff) is practically static. Since the final mass template 
        is normalized by forcing dN/dm to integrate to 1, any constant term 
        in this window cancels out mathematically.
        
        Args:
            p0 (float): True rigidity (P0) in GV.
            
        Returns:
            float: Multiplicative acceptance and exposure factor (dimensionless).
        """
        if p0 <= 0.0:
            return 0.0
            
        return 1.0

    def sigma_beta_resolution(self, beta0: float, e_kn: float, detector: str, cos_theta: float = 0.91) -> float:
        """
        Calculates the expected absolute standard deviation (sigma_beta) for the velocity 
        measurement according to the empirical parameterizations of the AMS-02.
        
        Analytical Approximation (Dimensionality Reduction):
        The value cos_theta = 0.91 represents the average value of the zenith angle of 
        incidence extracted from the Monte Carlo simulation. By fixing this value, 
        we avoid an extra and heavy integration in the three-dimensional spatial domain.
        
        Args:
            beta0 (float): Proposed theoretical velocity (v/c).
            e_kn (float): Theoretical kinetic energy per nucleon [GeV/n].
            detector (str): Evaluated subsystem ('TOF', 'NaF', or 'AGL').
            cos_theta (float): Incidence angle to calculate the effective distance d.
            
        Returns:
            float: Absolute error sigma_beta. Returns -1.0 if the particle 
                   fails the Cherenkov threshold or the detector is invalid.
        """
        if detector == 'TOF':
            # Temporal resolution sigma_t (in ns)
            sigma_t_ns = np.sqrt((self.tof_A_ps / self.z_charge)**2 + self.tof_B_ps**2) / 1000.0
            
            # Effective traversal distance and true time of flight
            d_cm = self.tof_z_dist / cos_theta
            t0_ns = d_cm / (beta0 * self.c_speed)
            
            # Error propagation (1st order Taylor): sigma_beta = beta^2 * (c * sigma_t / d)
            # Organized in relative terms for algorithmic readability
            sigma_beta_rel = beta0 * (sigma_t_ns / t0_ns)
            return sigma_beta_rel * beta0
            
        elif detector == 'NaF':
            # Particle below the threshold does not produce Cherenkov light
            if beta0 <= self.rich_naf_thr:
                return -1.0 
                
            # Empirical formula of the RICH resolution
            sigma_beta_rel = self.rich_naf_A - self.rich_naf_B * np.exp(1.0 - (e_kn / self.rich_naf_E0))
            return sigma_beta_rel * beta0
            
        elif detector == 'AGL':
            # Particle below the threshold does not produce Cherenkov light
            if beta0 <= self.rich_agl_thr:
                return -1.0 
                
            # Empirical formula of the RICH resolution
            sigma_beta_rel = self.rich_agl_A - self.rich_agl_B * np.exp(1.0 - (e_kn / self.rich_agl_E0))
            return sigma_beta_rel * beta0
            
        # Protection against invalid arguments
        return -1.0

    def evaluate_beta_pdf(self, beta: float, beta0: float, e_kn: float, detector: str, cos_theta: float = 0.91) -> float:
        """
        Calculates the value of the theoretical Probability Density Function (PDF) g(beta).
        
        For the RICH (NaF and AGL): 
        The resolution is calculated directly in velocity (beta). It is assumed that the 
        reconstructed angular error of the Cherenkov cone translates into a Normal 
        distribution around the true velocity (beta0).
        Formula: PDF(beta) = Normal(beta | mean=beta0, std=sigma_beta)
        
        For the TOF:
        The intrinsic measurement of the detector occurs in the time domain (t). 
        Since velocity is inversely proportional to time (beta = d / (c * t)), 
        a Normal distribution in time generates an asymmetric distribution in beta.
        Mathematical rigor requires evaluating the Gaussian in time and multiplying by the 
        Jacobian of the transformation to obtain the density in beta.
        
        TOF Formulas:
        - t = d / (c * beta)
        - Jacobian |dt/dbeta| = d / (c * beta^2)
        - PDF(beta) = Normal(t | mean=t_true, std=sigma_t) * |dt/dbeta|
        
        Args:
            beta (float): Measured velocity to evaluate.
            beta0 (float): True theoretical velocity of the particle.
            e_kn (float): Kinetic energy per nucleon [GeV/n] (required for the RICH).
            detector (str): Evaluated subsystem ('TOF', 'NaF', or 'AGL').
            cos_theta (float): Cosine of the incidence angle (fixed at 0.91).
            
        Returns:
            float: Probability density g(beta).
        """
        # Protection to avoid computational stresses with null denominators
        if beta <= 0.0:
            return 0.0

        if detector == 'TOF':
            # Temporal resolution sigma_t (in ns)
            sigma_t_ns = np.sqrt((self.tof_A_ps / self.z_charge)**2 + self.tof_B_ps**2) / 1000.0
            d_cm = self.tof_z_dist / cos_theta
            
            # Conversion of velocities to the true and measured time domains
            t_true = d_cm / (beta0 * self.c_speed)
            t_recon = d_cm / (beta * self.c_speed)
            
            # Calculation of the absolute Jacobian of the transformation |dt/dbeta|
            jacobian_t = d_cm / (self.c_speed * (beta**2))
            
            # Rigorous PDF evaluation
            return norm.pdf(t_recon, loc=t_true, scale=sigma_t_ns) * jacobian_t
            
        else:
            # RICH cases (NaF and AGL): evaluation in the velocity domain
            
            # Get parameterized absolute error sigma_beta
            sigma_b = self.sigma_beta_resolution(beta0, e_kn, detector, cos_theta)
            
            # If the particle does not emit light (sigma_b = -1.0)
            if sigma_b <= 0.0:
                return 0.0
                
            # Gaussian evaluation
            return norm.pdf(beta, loc=beta0, scale=sigma_b)

    def integrand_1D(self, beta: float, m_rec: float, m_true: float, detector: str) -> float:
        """
        Calculates the value of the differential integrand assuming an IDEAL Tracker (no error).
        
        Since the Tracker is perfect, the measured rigidity (P) equals the true one (P0).
        This reduces the double convolution to a simple integral only over beta.
        
        Kinematic relations (natural units c=1):
        - gamma = 1 / np.sqrt(1 - beta^2)
        - P0 = P = (m_rec * gamma * beta) / Z
        
        Change of Variables (Jacobian):
        To transition from the measured phase space (P, beta) to the (m, beta) space,
        multiply by the determinant of the Jacobian |dP/dm|:
        |J| = (gamma * beta) / Z
        
        Args:
            beta (float): Integration velocity in the measured space.
            m_rec (float): Reconstructed mass evaluated on the X-axis of the plot.
            m_true (float): Exact mass of the isotope to simulate (Monte Carlo truth).
            detector (str): Evaluated subsystem ('TOF', 'NaF', 'AGL').
            
        Returns:
            float: Differential value of the probability dN/dm for this point.
        """
        # The velocity must respect relativity
        if beta <= 0.0 or beta >= 0.999999:
            return 0.0
            
        # Reconstructed Kinematics
        gamma = 1.0 / np.sqrt(1.0 - beta**2)
        
        # Since the Tracker is ideal, P0 = P_reconstructed
        p0 = (m_rec * beta * gamma) / self.z_charge
        
        # True Kinematics (Monte Carlo Truth induced by p0)
        zp0 = self.z_charge * p0
        beta0 = zp0 / np.sqrt(zp0**2 + m_true**2)
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2)
        e_kn = (gamma0 - 1.0) * self.mass_nucleon
        
        # Resolution Function g(beta)
        g_beta = self.evaluate_beta_pdf(beta, beta0, e_kn, detector)
        if g_beta <= 0.0:
            return 0.0
        
        # A Priori Probability (Primary Flux and Acceptance)
        j_p0 = self.cosmic_flux(p0)
        g_p0 = self.acceptance_and_exposure(p0)
        
        # Transformation to the mass space
        jacobian = (gamma * beta) / self.z_charge
        
        return jacobian * j_p0 * g_p0 * g_beta

    def integrand_2D(self, p0: float, beta: float, m_rec: float, m_true: float, detector: str) -> float:
        """
        Calculates the complete integrand value for double convolution, including 
        the real Tracker uncertainty.
        
        Here, P0 (true) and P (measured) are independent variables. 
        The reconstructed mass is evaluated using the measured P. The 
        resolution function g_P(P ; P0) is added to the probability calculation.
        
        Args:
            p0 (float): True integration rigidity [GV].
            beta (float): Integration velocity.
            m_rec (float): Reconstructed mass to evaluate.
            m_true (float): Exact mass of the isotope.
            detector (str): Evaluated subsystem.
            
        Returns:
            float: Differential value d^2N / (dm d_beta) before the final integration.
        """
        if beta <= 0.0 or beta >= 0.999999 or p0 <= 0.0:
            return 0.0
            
        # Reconstructed Kinematics (Isolate the measured P from the mass)
        gamma = 1.0 / np.sqrt(1.0 - beta**2)
        p_measured = (m_rec * beta * gamma) / self.z_charge
        
        # True Kinematics 
        zp0 = self.z_charge * p0
        beta0 = zp0 / np.sqrt(zp0**2 + m_true**2)
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2)
        e_kn = (gamma0 - 1.0) * self.mass_nucleon
        
        # Tracker Resolution Function: g_P(P_measured ; P0)
        # CURRENT: Constant resolution. FUTURE WORK: Dependence on the MDR.
        sigma_p = p0 * self.tracker_sigma_rel
        g_P = norm.pdf(p_measured, loc=p0, scale=sigma_p)
        
        # If the tracker probability is zero, abort the rest of the calculations
        if g_P <= 0.0:
            return 0.0
        
        # Velocity Resolution Function: g_beta(beta ; beta0)
        g_beta = self.evaluate_beta_pdf(beta, beta0, e_kn, detector)
        if g_beta <= 0.0:
            return 0.0
        
        # A Priori Probability
        j_p0 = self.cosmic_flux(p0)
        g_p0 = self.acceptance_and_exposure(p0)
        
        # Change of Variables
        jacobian = (gamma * beta) / self.z_charge
        
        return jacobian * j_p0 * g_p0 * g_P * g_beta

    def _compute_single_bin(self, m_rec: float, m_true: float, beta_bin: tuple, detector: str, use_tracker_resolution: bool) -> float:
        """
        Calculates the integral (the total probability dN/dm) for a single point 
        of the mass plot.
        
        If the Tracker is ideal (use_tracker_resolution=False):
        Solves a 1D integral along the velocity window (beta_min to beta_max).
        
        If the Tracker has error (use_tracker_resolution=True):
        Solves a double integral. To save CPU, the integration over P0 does not go 
        from 0 to infinity, but focuses on the high probability zone around 
        the approximate momentum of the particle (p_approx).
        
        Args:
            m_rec (float): Evaluated reconstructed mass.
            m_true (float): Exact mass of the isotope.
            beta_bin (tuple): Velocity limits (beta_min, beta_max).
            detector (str): Evaluated subsystem.
            use_tracker_resolution (bool): Flag to activate the Tracker error.
            
        Returns:
            float: Area value of the probability for this specific mass.
        """
        beta_min, beta_max = beta_bin
        
        if not use_tracker_resolution:
            # Fast Method: 1D Integral only over d_beta
            # limit=200 and epsrel=1e-3 (0.1% error) optimize CPU time 
            # while maintaining perfectly adequate precision for the AMS-02 resolution
            integral, _ = integrate.quad(
                self.integrand_1D, 
                beta_min, 
                beta_max, 
                args=(m_rec, m_true, detector),
                limit=200, 
                epsrel=1e-3
            )
        else:
            # Complete Method: Double Integral d_P0 d_beta
            # Smart definition of the momentum integration limits (P0)
            # to prevent SciPy from integrating in zero-probability zones
            gamma_approx = 1.0 / np.sqrt(1.0 - beta_min**2)
            p_approx = (m_rec * beta_min * gamma_approx) / self.z_charge
            
            # OPTIMIZATION: Limit the integration window to +/- 5 sigmas of the Tracker
            # Reduces CPU time and prevents dblquad failures
            sigma_p_approx = p_approx * self.tracker_sigma_rel
            p0_min = max(0.1, p_approx - 5.0 * sigma_p_approx) 
            p0_max = p_approx + 5.0 * sigma_p_approx
            
            integral, _ = integrate.dblquad(
                self.integrand_2D, 
                beta_min, 
                beta_max, 
                lambda x: p0_min, 
                lambda x: p0_max,
                args=(m_rec, m_true, detector),
                epsrel=1e-3
            )
            
        return integral

    def compute_template(self, m_array: np.ndarray, m_true: float, beta_bin: tuple, 
                         detector: str, use_tracker_resolution: bool = False) -> np.ndarray:
        """
        Generates the normalized distribution dN/dm for an array of reconstructed masses.
        
        This is the high-level function that distributes the computational load across 
        all processor cores using parallel multiprocessing.
        In the end, it ensures that the template meets the fundamental physical condition of 
        probability conservation: the area under the curve must be equal to 1.
        
        Args:
            m_array (np.ndarray): Array with the points of the mass axis (X).
            m_true (float): Exact mass of the isotope to simulate.
            beta_bin (tuple): Velocity window.
            detector (str): Evaluated subsystem.
            use_tracker_resolution (bool): Activates convolution with the Tracker error.
            
        Returns:
            np.ndarray: Array with the normalized probability values (Y-axis).
        """
        beta_min, beta_max = beta_bin
        
        print(f"[{detector}] Calculating Analytical Template (Tracker Error = {use_tracker_resolution})")
        print(f"[{detector}] True Mass: {m_true:.4f} | Velocity Bin: [{beta_min:.3f}, {beta_max:.3f}]")
        print(f"[{detector}] Initializing parallel computation for {len(m_array)} mass bins...")
        
        # The partial module freezes the static parameters of _compute_single_bin, 
        # leaving only the iterable variable (m_rec) exposed for mapping
        worker_function = partial(self._compute_single_bin,
                                  m_true=m_true, 
                                  beta_bin=beta_bin, 
                                  detector=detector, 
                                  use_tracker_resolution=use_tracker_resolution)
            
        # The ProcessPoolExecutor requests the logical cores of the computer.
        # The map method spreads the function and returns the results in the same order as m_array
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(worker_function, m_array))
            
        template_values = np.array(results, dtype=float)
            
        # Template Normalization: the total probability integrates to 1
        # The trapezoid method approximates the discrete area under the curve
        total_area = np.trapezoid(template_values, m_array)
        
        if total_area > 0.0:
            template_values /= total_area
        else:
            print(f"[{detector}] WARNING: The calculated total area of the template is zero or negative!")
            
        return template_values

# ===================================================================
# INDEPENDENT EXECUTION TEST
# ===================================================================
if __name__ == "__main__":
    import time
    
    print("=" * 50)
    print(" AMS-02 ANALYTICAL ENGINE TEST")
    print("=" * 50)
    
    start_time = time.time()
    template_engine = AMSMassTemplateAnalytic()
    
    # Example: Moderate velocity bin where the TOF has good response
    test_beta_bin = (0.85, 0.90) 
    
    # Create a mass array (X-axis) from 8 to 13 GeV/c^2 (100 points)
    masses_x = np.linspace(8.0, 13.0, 100)
    
    # Generate model for Boron-10 
    b10_distribution = template_engine.compute_template(
        m_array=masses_x, 
        m_true=MASS_B10, 
        beta_bin=test_beta_bin, 
        detector='TOF', 
        use_tracker_resolution=True
    )
    
    end_time = time.time()
    
    print("\n" + "=" * 50)
    print(" TEST RESULTS")
    print("=" * 50)
    print(f" Processing time        : {end_time - start_time:.2f} seconds")
    print(f" Calculated points      : {len(b10_distribution)}")
    print(f" Maximum curve value    : {np.max(b10_distribution):.4f}")
    
    # Physical sanity check (The total probability area must be 1)
    test_area = np.trapezoid(b10_distribution, masses_x)
    print(f" Area under the curve   : {test_area:.4f} (Should be 1.0000)")
    print("=" * 50)
    print(" Processing completed. The mathematical engine is operational.")

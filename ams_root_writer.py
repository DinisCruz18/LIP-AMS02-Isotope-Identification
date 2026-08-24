import ROOT
from array import array
import random
import numpy as np

class AMSRootWriter:
    """
    Class responsible for initializing, allocating memory, and filling 
    a ROOT Tree with the Monte Carlo simulation data of the AMS-02.
    
    This structure simulates the 'raw' data, storing 
    the kinematic and instrumental variables required for the subsequent 
    isotopic separation analysis (10B vs 11B). Only events that 
    pass the geometric trigger (TOF acceptance) are recorded.
    """
    
    def __init__(self, filename="ams_isotopes.root"):
        """
        Initializes the ROOT file and allocates the C memory pointers 
        that will bridge Python and the TTree 'Branches'.
        """
        self.file = ROOT.TFile(filename, "RECREATE")
        self.tree = ROOT.TTree("tree_ams", "MC Data of Boron Isotopes")
        
        # ===================================================================
        # 1. MEMORY ALLOCATION (Arrays of size 1 for C++ pointers)
        # ===================================================================
        
        # --- Identification and Weight ---
        self.Z = array('i', [0])               # Electric charge
        self.A = array('i', [0])               # Isotope mass number (10 or 11)
        self.w = array('f', [0.0])             # Statistical weight (flux correction)
        
        # --- Generation Kinematics (Upper LTOF geometry) ---
        self.x = array('f', [0.0])             # X impact position [cm]
        self.y = array('f', [0.0])             # Y impact position [cm]
        self.cos_theta = array('f', [0.0])     # Cosine of the zenith angle
        self.phi = array('f', [0.0])           # Azimuthal angle [rad]
        
        # --- True Quantities ---
        self.P0 = array('f', [0.0])            # Generated true rigidity [GV]
        self.e_kn = array('f', [0.0])          # True kinetic energy per nucleon [GeV/n]
        self.T0 = array('f', [0.0])            # True total kinetic energy [GeV]
        self.beta_0 = array('f', [0.0])        # True velocity [v/c]
        self.m_true = array('f', [0.0])        # Exact rest mass [GeV/c^2]
        
        # --- Reconstructed Quantities (With detector resolution) ---
        self.P = array('f', [0.0])             # Rigidity measured in the Tracker [GV]
        
        self.beta_TOF = array('f', [0.0])      # Velocity measured in the TOF [v/c]
        self.radiator = array('i', [0])        # RICH Radiator (0=None, 1=NaF, 2=AGL)
        self.beta_RICH = array('f', [0.0])     # Velocity measured in the RICH [v/c]
        
        self.T_medido_TOF = array('f', [0.0])  # Energy reconstructed via TOF [GeV]
        self.T_medido_RICH = array('f', [0.0]) # Energy reconstructed via RICH [GeV]
        
        self.m_TOF = array('f', [0.0])         # Reconstructed mass (Tracker + TOF) [GeV/c^2]
        self.m_RICH = array('f', [0.0])        # Reconstructed mass (Tracker + RICH) [GeV/c^2]

        # ===================================================================
        # 2. CREATION OF ROOT TREE BRANCHES
        # ===================================================================
        self.tree.Branch("Z", self.Z, "Z/I")
        self.tree.Branch("A", self.A, "A/I")
        self.tree.Branch("w", self.w, "w/F")
        self.tree.Branch("x", self.x, "x/F")
        self.tree.Branch("y", self.y, "y/F")
        self.tree.Branch("cos_theta", self.cos_theta, "cos_theta/F")
        self.tree.Branch("phi", self.phi, "phi/F")
        
        self.tree.Branch("P0", self.P0, "P0/F")
        self.tree.Branch("e_kn", self.e_kn, "e_kn/F")
        self.tree.Branch("T0", self.T0, "T0/F") 
        self.tree.Branch("beta_0", self.beta_0, "beta_0/F")
        self.tree.Branch("m_true", self.m_true, "m_true/F")
        
        self.tree.Branch("P", self.P, "P/F")
        self.tree.Branch("beta_TOF", self.beta_TOF, "beta_TOF/F")
        self.tree.Branch("radiator", self.radiator, "radiator/I")
        self.tree.Branch("beta_RICH", self.beta_RICH, "beta_RICH/F")
        
        self.tree.Branch("T_medido_TOF", self.T_medido_TOF, "T_medido_TOF/F")
        self.tree.Branch("T_medido_RICH", self.T_medido_RICH, "T_medido_RICH/F")
        self.tree.Branch("m_TOF", self.m_TOF, "m_TOF/F")
        self.tree.Branch("m_RICH", self.m_RICH, "m_RICH/F")

    def fill_event(self, event_data: dict):
        """
        Processes the dictionary of a kinematic event and fills the ROOT Tree.
        
        In experimental physics, when a detector fails to reconstruct an 
        observable (e.g., the particle misses the RICH radiator or does not reach the 
        emission threshold), a negative numerical flag (-999.0) is injected. 
        This prevents instrumental failures from accumulating at the value 0.0, 
        corrupting the statistical analysis of the histograms.
        
        Args:
            event_data (dict): Dictionary containing the generated and measured quantities.
        """
        # ===================================================================
        # 1. IDENTIFICATION AND WEIGHT
        # ===================================================================
        self.Z[0] = event_data['Z']
        self.A[0] = event_data['A']
        self.w[0] = event_data['w']
        
        # ===================================================================
        # 2. GENERATION KINEMATICS
        # ===================================================================
        self.x[0] = event_data['x']
        self.y[0] = event_data['y']
        self.cos_theta[0] = event_data['cos_theta']
        self.phi[0] = event_data['phi']
        
        # ===================================================================
        # 3. TRUE QUANTITIES
        # ===================================================================
        self.P0[0] = event_data['P0']
        self.e_kn[0] = event_data['e_kn']
        self.T0[0] = event_data['T0']
        self.beta_0[0] = event_data['beta_0']
        self.m_true[0] = event_data['m_true']
        
        # ===================================================================
        # 4. MEASURED AND RECONSTRUCTED QUANTITIES
        # ===================================================================
        self.P[0] = event_data['P'] 
        self.beta_TOF[0] = event_data['beta_TOF']
        
        # Efficient conversion of the radiator flag to an integer index (ROOT)
        rad_str = event_data.get('Radiator')
        if rad_str == 'NaF':
            self.radiator[0] = 1
        elif rad_str == 'AGL':
            self.radiator[0] = 2
        else:
            self.radiator[0] = 0  # 0 indicates geometric acceptance failure in the RICH
            
        # Protection against reconstruction failures using the HEP flag (-999.0)
        self.beta_RICH[0] = event_data.get('beta_RICH') if event_data.get('beta_RICH') is not None else -999.0
        
        self.T_medido_TOF[0] = event_data.get('T_medido_TOF') if event_data.get('T_medido_TOF') is not None else -999.0
        self.T_medido_RICH[0] = event_data.get('T_medido_RICH') if event_data.get('T_medido_RICH') is not None else -999.0
        
        self.m_TOF[0] = event_data.get('m_TOF') if event_data.get('m_TOF') is not None else -999.0
        self.m_RICH[0] = event_data.get('m_RICH') if event_data.get('m_RICH') is not None else -999.0
        
        # ===================================================================
        # 5. TREE FILLING
        # ===================================================================
        self.tree.Fill()

    def close(self):
        """
        Writes the data from the intermediate memory (buffer) to the hard drive 
        and safely closes the ROOT file.
        
        Interrupting the process before 
        the file is closed may result in a corrupted TTree.
        """
        self.file.cd()
        self.tree.Write()
        self.file.Close()
        print("ROOT Tree saved and file closed successfully.")

# ===================================================================
# UNIT TEST MODULE 
# ===================================================================
if __name__ == "__main__":
    print("Initializing the ROOT writer for unit testing...")
    writer = AMSRootWriter("ams_test_data.root")
    
    n_eventos = 5000
    print(f"Generating {n_eventos} mock events (Mock Data)...")
    
    for i in range(n_eventos):
        # Realistic failure logic: simulates the particle not hitting the radiators
        radiator_choice = random.choice(['NaF', 'AGL', None])
        is_rich_valid = radiator_choice is not None
        
        mock_event = {
            # Identification and Generation
            "Z": 5,
            "A": random.choice([10, 11]),
            "w": random.uniform(0.1, 1.5),
            
            "x": random.uniform(-65, 65),
            "y": random.uniform(-65, 65),
            "cos_theta": random.uniform(0.1, 1.0), # cos(theta) > 0 to pass the TOF
            "phi": random.uniform(0, 2 * np.pi),
            
            # True Quantities
            "P0": random.uniform(1.0, 100.0),
            "e_kn": random.uniform(0.5, 10.0),
            "T0": random.uniform(0.1, 10.0), 
            "beta_0": random.uniform(0.8, 0.99),
            "m_true": random.choice([10.0129, 11.0093]),
            
            # Tracker and TOF (Always reconstructed in an event that passes the trigger)
            "P": random.uniform(1.0, 110.0),
            "beta_TOF": random.uniform(0.79, 1.0),
            "T_medido_TOF": random.uniform(0.1, 10.0),
            "m_TOF": random.uniform(8.0, 13.0),
            
            # RICH (Actively tests the injection of the HEP flag -999.0 if is_rich_valid is false)
            "Radiator": radiator_choice,
            "beta_RICH": random.uniform(0.85, 1.0) if is_rich_valid else None,
            "T_medido_RICH": random.uniform(0.1, 10.0) if is_rich_valid else None,
            "m_RICH": random.uniform(9.0, 12.0) if is_rich_valid else None
        }
        
        writer.fill_event(mock_event)
        
    writer.close()
    print("Test completed.")
"""
====================================================================================
AMS-02 1D DIAGNOSTICS
====================================================================================
Script for generating 1D sanity checks and kinematic distributions.
Outputs clean, publication-ready plots with statistical boxes.
"""

import os
import ROOT
import numpy as np
from array import array

def setup_environment(output_dir="plots_basic"):
    """Configures the ROOT environment for publication-quality 1D plots."""
    ROOT.gROOT.SetBatch(True)
    
    # Enable stat box with: Name, Entries, Mean, StdDev
    ROOT.gStyle.SetOptStat(1110)
    
    # Aesthetic formatting for the stat box
    ROOT.gStyle.SetStatBorderSize(1)
    ROOT.gStyle.SetStatX(0.95) # Slightly offset from the right edge
    ROOT.gStyle.SetStatY(0.92) # Slightly offset from the top edge
    ROOT.gStyle.SetStatW(0.20)
    ROOT.gStyle.SetStatH(0.15)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")
        
    return output_dir

def create_log_bins(n_bins: int, min_val: float, max_val: float):
    """Generates an array of logarithmic bin edges."""
    edges = np.logspace(np.log10(min_val), np.log10(max_val), n_bins + 1)
    return array('d', edges)

def generate_1d_histograms(tree, output_dir):
    """
    Iterates over a list of configurations to generate 1D histograms 
    for each specified ROOT Tree branch with proper binning and scales.
    """
    # Configuration dictionary per physical variable
    # Titles restored to format: "Main Title;X-Axis;Y-Axis"
    plot_configs = [
        # --- Discrete Variables ---
        {"var": "Z", "title": "Charge Distribution;Charge Z;Events / Bin", "bins": 25, "min": 2.5, "max": 7.5, "logx": False, "logy": False},
        {"var": "A", "title": "Isotopic Mass Number;Isotopic Mass Number A;Events / Bin", "bins": 25, "min": 8.5, "max": 13.5, "logx": False, "logy": False},
        
        # --- Spatial and Angular Variables ---
        {"var": "x", "title": "Initial X Position;Initial Position x [cm];Events / Bin", "bins": 130, "min": -70.0, "max": 70.0, "logx": False, "logy": False},
        {"var": "y", "title": "Initial Y Position;Initial Position y [cm];Events / Bin", "bins": 130, "min": -70.0, "max": 70.0, "logx": False, "logy": False},
        {"var": "cos_theta", "title": "Zenith Angle Cosine;Zenith Angle Cosine cos(#theta);Events / Bin", "bins": 100, "min": 0.0, "max": 1.0, "logx": False, "logy": False},
        {"var": "theta", "branch": "cos_theta", "expr": "TMath::ACos(cos_theta)", "title": "Zenith Angle;Zenith Angle #theta [rad];Events / Bin", "bins": 100, "min": 0.0, "max": np.pi/2, "logx": False, "logy": False},
        {"var": "phi", "title": "Azimuthal Angle;Azimuthal Angle #phi [rad];Events / Bin", "bins": 100, "min": 0.0, "max": 2*np.pi, "logx": False, "logy": False},
        
        # --- Kinematic and Flux Variables ---
        {"var": "P0", "title": "Generated Rigidity;Generated Rigidity P_{0} [GV];Events / Bin", "bins": 100, "min": 1.0, "max": 1000.0, "logx": True, "logy": True},
        {"var": "P", "title": "Measured Rigidity;Measured Rigidity P [GV];Events / Bin", "bins": 100, "min": 1.0, "max": 1000.0, "logx": True, "logy": True},
        {"var": "e_kn", "title": "Kinetic Energy per Nucleon;Kinetic Energy per Nucleon E_{kn} [GeV/n];Events / Bin", "bins": 100, "min": 0.1, "max": 100.0, "logx": True, "logy": True},
        {"var": "T0", "title": "Total Kinetic Energy;Total Kinetic Energy T_{0} [GeV];Events / Bin", "bins": 100, "min": 0.1, "max": 5000.0, "logx": True, "logy": True},
        {"var": "w", "title": "Statistical Weight;Statistical Weight w;Events / Bin", "bins": 100, "min": 1e-3, "max": 5000.0, "logx": True, "logy": True},
        
        # --- Velocities ---
        {"var": "beta_0", "title": "True Velocity;True Velocity #beta_{0};Events / Bin", "bins": 200, "min": 0.4, "max": 1.05, "logx": False, "logy": True},
        {"var": "beta_TOF", "title": "Measured Velocity (TOF);Measured Velocity #beta_{TOF};Events / Bin", "bins": 200, "min": 0.4, "max": 1.05, "logx": False, "logy": True},
        {"var": "beta_RICH", "title": "Measured Velocity (RICH);Measured Velocity #beta_{RICH};Events / Bin", "bins": 200, "min": 0.8, "max": 1.05, "logx": False, "logy": True},

        # --- Measured Energies ---
        {"var": "T_measured_TOF", "title": "Measured Kinetic Energy (TOF);Measured Kinetic Energy T_{TOF} [GeV];Events / Bin", "bins": 100, "min": 0.1, "max": 5000.0, "logx": True, "logy": True},
        {"var": "T_measured_RICH", "title": "Measured Kinetic Energy (RICH);Measured Kinetic Energy T_{RICH} [GeV];Events / Bin", "bins": 100, "min": 0.1, "max": 5000.0, "logx": True, "logy": True},

        # --- Masses ---
        {"var": "m_true", "title": "True Isotopic Mass;True Mass m_{true} [GeV/c^{2}];Events / Bin", "bins": 50, "min": 8.0, "max": 13.0, "logx": False, "logy": False},
        {"var": "m_TOF", "title": "Reconstructed Mass (TOF);Reconstructed Mass m_{TOF} [GeV/c^{2}];Events / Bin", "bins": 150, "min": 4.0, "max": 20.0, "logx": False, "logy": False},
        {"var": "m_RICH", "title": "Reconstructed Mass (RICH);Reconstructed Mass m_{RICH} [GeV/c^{2}];Events / Bin", "bins": 150, "min": 4.0, "max": 20.0, "logx": False, "logy": False},
        
        # --- Velocity Residuals (Measured - True) ---
        {"var": "res_beta_TOF", "branch": "beta_TOF", "expr": "beta_TOF - beta_0", "title": "Velocity Residual (TOF);Velocity Residual #Delta#beta_{TOF};Events / Bin", "bins": 150, "min": -0.15, "max": 0.15, "logx": False, "logy": True},
        {"var": "res_beta_RICH", "branch": "beta_RICH", "expr": "beta_RICH - beta_0", "title": "Velocity Residual (RICH);Velocity Residual #Delta#beta_{RICH};Events / Bin", "bins": 150, "min": -0.02, "max": 0.02, "logx": False, "logy": True},
    ]

    for config in plot_configs:
        var_name = config["var"]
        
        # Use explicit values if provided, otherwise default to var_name
        branch_name = config.get("branch", var_name)
        draw_expr = config.get("expr", branch_name)
        
        # Verify if the base branch exists in the Tree
        if not tree.GetBranch(branch_name):
            print(f"[WARNING] Branch '{branch_name}' not found. Skipping {var_name}...")
            continue
            
        c = ROOT.TCanvas(f"c_{var_name}", f"Distribution of {var_name}", 800, 600)
        
        # Margin adjustments: Left margin increased to prevent Y-axis title overlap
        c.SetBottomMargin(0.12)
        c.SetLeftMargin(0.15) 
        c.SetRightMargin(0.05) 
        
        # Configure logarithmic axes
        if config["logx"]: c.SetLogx()
        if config["logy"]: c.SetLogy()
        
        # Histogram creation
        if config["logx"]:
            bins_array = create_log_bins(config["bins"], config["min"], config["max"])
            hist = ROOT.TH1F(f"h_{var_name}", config["title"], config["bins"], bins_array)
        else:
            hist = ROOT.TH1F(f"h_{var_name}", config["title"], config["bins"], config["min"], config["max"])
            
        # Scientific aesthetics
        hist.SetLineColor(ROOT.kBlue + 2)
        hist.SetLineWidth(2)
        hist.SetFillColorAlpha(ROOT.kBlue + 1, 0.1)
        
        # Prevent title overlap (Increased Y-offset)
        hist.GetXaxis().SetTitleOffset(1.2)
        hist.GetYaxis().SetTitleOffset(1.6)
        
        # Apply drawing expression and filter instrumental failures (-999.0)
        cut = f"{branch_name} != -999.0"
        tree.Draw(f"{draw_expr} >> h_{var_name}", cut, "HIST")
        
        # Y-axis adjustment for logarithmic scales
        if config["logy"]:
            hist.SetMinimum(0.5)
            
        save_path = os.path.join(output_dir, f"{var_name}_dist.pdf")
        c.SaveAs(save_path)
        
    print("-" * 60)
    print(f" Processing complete. 1D plots saved in '{output_dir}/'.")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print(" AMS-02 1D DIAGNOSTICS PIPELINE")
    print("=" * 60)
    
    out_dir = setup_environment()
    
    file_name = "ams_b10_b11_simulation.root"
    if not os.path.exists(file_name):
        print(f"[ERROR] File '{file_name}' not found.")
        exit(1)
        
    print(f"[INFO] Opening file: {file_name}")
    root_file = ROOT.TFile(file_name, "READ")
    data_tree = root_file.Get("tree_ams")
    
    print("-" * 60)
    print(" Generating 1D histograms...")
    print("-" * 60)
    
    generate_1d_histograms(data_tree, out_dir)
    root_file.Close()
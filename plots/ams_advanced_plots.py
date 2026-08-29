"""
====================================================================================
AMS-02 ADVANCED PLOTTING AND DIAGNOSTICS
====================================================================================
Unified plotting script for the AMS-02 isotopic separation pipeline.
This module generates production-ready, publication-quality 1D and 2D ROOT plots 
for kinematics, geometric acceptance, instrumental resolution, and mass separation.
"""

import os
import ROOT
import numpy as np
from array import array

# ===================================================================
# 0. CORE UTILITIES
# ===================================================================

def setup_environment(output_dir="plots_advanced"):
    """
    Configures the global ROOT environment for scientific plotting.
    """
    ROOT.gROOT.SetBatch(True)            
    ROOT.gStyle.SetOptStat(0)            
    
    # Modern scientific standard (Viridis). 
    # Much more perceptually uniform and professional than the old Rainbow.
    ROOT.gStyle.SetPalette(ROOT.kBird) 
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")
        
    return output_dir

def create_log_bins(n_bins: int, min_val: float, max_val: float) -> array:
    edges = np.logspace(np.log10(min_val), np.log10(max_val), n_bins + 1)
    return array('d', edges)

# ===================================================================
# BLOCK A: KINEMATICS & GENERATION
# ===================================================================

def plot_kinematic_spectrum(tree, output_dir):
    """
    Validates the statistical weighting (w).
    Compares the log-uniform generation with the reweighted physical flux.
    """
    c1 = ROOT.TCanvas("c1_kinematics", "Kinematic Spectrum", 800, 600)
    c1.SetLogx()
    c1.SetLogy()
    
    # Increase bottom margin to make room for the X-axis title
    c1.SetBottomMargin(0.12)
    
    bins_p = create_log_bins(100, 1.0, 1000.0)
    
    h_unweighted = ROOT.TH1F("h_unweighted", "Kinematic Spectrum (Generation);Generated Rigidity P_{0} [GV];Events / Bin", 100, bins_p)
    h_unweighted.Sumw2()
    h_unweighted.SetLineColor(ROOT.kGray + 1)
    h_unweighted.SetLineWidth(2)
    
    # Push the X-axis title down to avoid overlapping the numbers
    h_unweighted.GetXaxis().SetTitleOffset(1.3)
    
    h_weighted = ROOT.TH1F("h_weighted", "Kinematic Spectrum (Physical Flux);Generated Rigidity P_{0} [GV];Events / Bin", 100, bins_p)
    h_weighted.Sumw2()
    h_weighted.SetLineColor(ROOT.kBlue + 2)
    h_weighted.SetLineWidth(2)
    
    cut_trigger = "beta_TOF != -999.0" 
    weight_cut = f"w * ({cut_trigger})"

    tree.Draw("P0 >> h_unweighted", cut_trigger, "goff")
    tree.Draw("P0 >> h_weighted", weight_cut, "goff")
    
    max_y = max(h_unweighted.GetMaximum(), h_weighted.GetMaximum())
    h_unweighted.SetMaximum(max_y * 10.0)
    h_unweighted.SetMinimum(1.0) 
    
    h_unweighted.Draw("HIST")
    h_weighted.Draw("HIST SAME")
    
    leg = ROOT.TLegend(0.5, 0.75, 0.88, 0.88)
    leg.SetBorderSize(0) 
    leg.AddEntry(h_unweighted, "Flat Generation (Unweighted)", "l")
    leg.AddEntry(h_weighted, "Physical Flux (Weighted)", "l")
    leg.Draw()
    
    save_path = os.path.join(output_dir, "A1_kinematic_spectrum.pdf")
    c1.SaveAs(save_path)
    print(f"[INFO] Saved: {save_path}")

def plot_kinetic_energy_correlation(tree, output_dir):
    """
    Compares the reconstructed kinetic energy (T_measured) from both TOF and RICH 
    against the generated true kinetic energy (T_0).
    """
    c = ROOT.TCanvas("c_t_corr", "Kinetic Energy Correlation", 1200, 500)
    c.Divide(2, 1)
    
    bins_t = create_log_bins(100, 0.1, 5000.0)
    
    expr_t_tof = "(1.0 / sqrt(1.0 - beta_TOF**2) - 1.0) * (A * 0.9315)"
    expr_t_rich = "(1.0 / sqrt(1.0 - beta_RICH**2) - 1.0) * (A * 0.9315)"
    
    cut_tof = "w * (beta_TOF != -999.0 && beta_TOF < 1.0)"
    cut_rich = "w * (beta_RICH != -999.0 && beta_RICH < 1.0)"
    
    # --- Panel 1: T_measured (TOF) vs T_0 ---
    c.cd(1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetRightMargin(0.18) 
    ROOT.gPad.SetBottomMargin(0.12) # Margin for X-axis title
    
    h_t_tof = ROOT.TH2F("h_t_tof", "Kinetic Energy Correlation (TOF);True Kinetic Energy T_{0} [GeV];Measured T_{TOF} [GeV]", 100, bins_t, 100, bins_t)
    h_t_tof.Sumw2()
    h_t_tof.GetXaxis().SetTitleOffset(1.3) # Prevent X-axis overlap
    h_t_tof.GetZaxis().SetTitle("Events / Bin")
    h_t_tof.GetZaxis().SetTitleOffset(1.4)
    
    tree.Draw(f"{expr_t_tof}:T0 >> h_t_tof", cut_tof, "COLZ")
    
    diag_tof = ROOT.TLine(0.1, 0.1, 5000.0, 5000.0)
    diag_tof.SetLineColor(ROOT.kRed)
    diag_tof.SetLineStyle(2)
    diag_tof.Draw("SAME")
    
    # --- Panel 2: T_measured (RICH) vs T_0 ---
    c.cd(2)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetRightMargin(0.18)
    ROOT.gPad.SetBottomMargin(0.12) # Margin for X-axis title
    
    h_t_rich = ROOT.TH2F("h_t_rich", "Kinetic Energy Correlation (RICH);True Kinetic Energy T_{0} [GeV];Measured T_{RICH} [GeV]", 100, bins_t, 100, bins_t)
    h_t_rich.Sumw2()
    h_t_rich.GetXaxis().SetTitleOffset(1.3) # Prevent X-axis overlap
    h_t_rich.GetZaxis().SetTitle("Events / Bin")
    h_t_rich.GetZaxis().SetTitleOffset(1.7)
    
    tree.Draw(f"{expr_t_rich}:T0 >> h_t_rich", cut_rich, "COLZ")
    
    diag_rich = ROOT.TLine(0.1, 0.1, 5000.0, 5000.0)
    diag_rich.SetLineColor(ROOT.kRed)
    diag_rich.SetLineStyle(2)
    diag_rich.Draw("SAME")
    
    save_path = os.path.join(output_dir, "A2_kinetic_energy_correlation.pdf")
    c.SaveAs(save_path)
    print(f"[INFO] Saved: {save_path}")

# ===================================================================
# BLOCK B: DETECTOR GEOMETRY
# ===================================================================

def plot_rich_radiator_map(tree, output_dir):
    """
    Visualizes the geometric acceptance of the RICH detector.
    Distinguishes the central NaF square and the outer AGL ring.
    """
    # Canvas quadrado
    c = ROOT.TCanvas("c_rich_rad", "RICH Radiator Map", 800, 800)
    
    # FORÇAR CÍRCULO PERFEITO: Margens exatamente iguais em todos os eixos
    margin = 0.15
    c.SetRightMargin(margin)
    c.SetLeftMargin(margin)
    c.SetTopMargin(margin)
    c.SetBottomMargin(margin)
    
    # 192 bins de -72 a 72 cm
    h_geo = ROOT.TProfile2D("h_geo", ";x_{RICH} [cm];y_{RICH} [cm]", 192, -72, 72, 192, -72, 72)
    
    cut_rich = "radiator == 1 || radiator == 2"
    z_rich = 134.0 
    tan_theta_expr = "sqrt(1 - cos_theta**2)/cos_theta"
    x_rich_expr = f"x - {z_rich} * ({tan_theta_expr}) * cos(phi)"
    y_rich_expr = f"y - {z_rich} * ({tan_theta_expr}) * sin(phi)"
    
    draw_expr = f"radiator:({y_rich_expr}):({x_rich_expr}) >> h_geo"
    
    tree.Draw(draw_expr, cut_rich, "goff")
    
    h_geo.SetMinimum(0.5)
    h_geo.SetMaximum(2.5)
    
    h_geo.GetXaxis().SetTitleOffset(1.2)
    h_geo.GetYaxis().SetTitleOffset(1.4)
    h_geo.GetZaxis().SetTitle("Radiator Index")
    h_geo.GetZaxis().SetTitleOffset(1.2)
    
    h_geo.Draw("COLZ")
    
    # CAIXA DE TEXTO: Movida para o canto extremo superior esquerdo (espaço branco)
    pave = ROOT.TPaveText(0.17, 0.75, 0.38, 0.83, "NDC")
    pave.SetFillColorAlpha(ROOT.kWhite, 0.9)
    pave.SetBorderSize(1) 
    pave.SetTextAlign(12) 
    pave.SetTextSize(0.035)
    pave.AddText("Index 1: NaF")
    pave.AddText("Index 2: AGL")
    pave.Draw("SAME")
    
    save_path = os.path.join(output_dir, "B1_rich_radiator_map.pdf")
    c.SaveAs(save_path)
    print(f"[INFO] Saved: {save_path}")

def plot_rich_spatial_density(tree, output_dir):
    """
    Projects the statistical event density (weighted counts) 
    onto the RICH radiator geometric plane.
    """
    c = ROOT.TCanvas("c_rich_dens", "RICH Spatial Density", 925, 800)
    # Slightly larger right margin to accommodate large numbers (e.g., 10000) on the Z-axis
    c.SetRightMargin(0.18) 
    c.SetLeftMargin(0.15)
    
    h_density = ROOT.TH2F("h_density", "RICH Spatial Event Density;x_{RICH} [cm];y_{RICH} [cm]", 192, -72, 72, 192, -72, 72)
    h_density.Sumw2()
    
    cut_rich = "radiator == 1 || radiator == 2"
    weight_cut = f"w * ({cut_rich})" 
    
    z_rich = 134.0 
    tan_theta_expr = "sqrt(1 - cos_theta**2)/cos_theta"
    x_rich_expr = f"x - {z_rich} * ({tan_theta_expr}) * cos(phi)"
    y_rich_expr = f"y - {z_rich} * ({tan_theta_expr}) * sin(phi)"
    
    draw_expr = f"({y_rich_expr}):({x_rich_expr}) >> h_density"
    
    tree.Draw(draw_expr, weight_cut, "goff")
    
    # Adjust aesthetics
    h_density.GetYaxis().SetTitleOffset(1.6)
    h_density.GetZaxis().SetTitle("Events / Bin")
    h_density.GetZaxis().SetTitleOffset(1.8)
    
    h_density.Draw("COLZ")
    
    save_path = os.path.join(output_dir, "B2_rich_spatial_density.pdf")
    c.SaveAs(save_path)
    print(f"[INFO] Saved: {save_path}")


# ===================================================================
# BLOCK C: INSTRUMENTAL RESOLUTION & SMEARING
# ===================================================================

def plot_tracker_migration(tree, output_dir):
    """
    Tracker Migration Matrix.
    Plots Measured Rigidity (P) vs True Rigidity (P0) to show resolution effects.
    """
    c = ROOT.TCanvas("c_tracker_mig", "Tracker Migration", 800, 600)
    c.SetLogx()
    c.SetLogy()
    
    # Margins to fit axis titles cleanly
    c.SetRightMargin(0.15)
    c.SetBottomMargin(0.12)
    
    bins_p = create_log_bins(100, 1.0, 1000.0)
    
    h_mig = ROOT.TH2F("h_mig", "Tracker Migration Matrix;True Rigidity P_{0} [GV];Measured Rigidity P [GV]", 100, bins_p, 100, bins_p)
    h_mig.Sumw2()
    
    # Aesthetic offsets
    h_mig.GetXaxis().SetTitleOffset(1.3)
    h_mig.GetYaxis().SetTitleOffset(1.3)
    h_mig.GetZaxis().SetTitle("Events / Bin")
    h_mig.GetZaxis().SetTitleOffset(1.3)
    
    # Fill the migration matrix applying statistical weights
    tree.Draw("P:P0 >> h_mig", "w", "COLZ")
    
    # Add ideal diagonal line (y=x) for visual reference
    diag = ROOT.TLine(1.0, 1.0, 1000.0, 1000.0)
    diag.SetLineColor(ROOT.kRed)
    diag.SetLineStyle(2)
    diag.Draw("SAME")
    
    save_path = os.path.join(output_dir, "C1_tracker_migration.pdf")
    c.SaveAs(save_path)
    print(f"[INFO] Saved: {save_path}")

def plot_velocity_smearing(tree, output_dir):
    """
    Validates the relativistic smearing.
    Plots true beta, beta_TOF, and beta_RICH against True Rigidity (P0).
    """
    c = ROOT.TCanvas("c_vel_smear", "Velocity Smearing", 1200, 400)
    c.Divide(3, 1) # Divide canvas into 3 horizontal panels
    
    bins_p = create_log_bins(100, 1.0, 1000.0)
    
    # --- Panel 1: True Velocity (No uncertainty) ---
    c.cd(1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetRightMargin(0.15)
    ROOT.gPad.SetBottomMargin(0.12)
    
    h_beta_true = ROOT.TH2F("h_beta_true", "Velocity Smearing (True);True Rigidity P_{0} [GV];True Velocity #beta_{0}", 100, bins_p, 100, 0.5, 1.05)
    h_beta_true.GetXaxis().SetTitleOffset(1.3)
    h_beta_true.GetZaxis().SetTitle("Events / Bin")
    h_beta_true.GetZaxis().SetTitleOffset(1.3)
    tree.Draw("beta_0:P0 >> h_beta_true", "w * (beta_TOF != -999.0)", "COLZ")
    
    # --- Panel 2: TOF Velocity (Lower resolution) ---
    c.cd(2)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetRightMargin(0.15)
    ROOT.gPad.SetBottomMargin(0.12)
    
    h_beta_tof = ROOT.TH2F("h_beta_tof", "Velocity Smearing (TOF);True Rigidity P_{0} [GV];Measured Velocity #beta_{TOF}", 100, bins_p, 100, 0.5, 1.05)
    h_beta_tof.GetXaxis().SetTitleOffset(1.3)
    h_beta_tof.GetZaxis().SetTitle("Events / Bin")
    h_beta_tof.GetZaxis().SetTitleOffset(1.3)
    tree.Draw("beta_TOF:P0 >> h_beta_tof", "w * (beta_TOF != -999.0)", "COLZ")
    
    # --- Panel 3: RICH Velocity (High resolution, with threshold) ---
    c.cd(3)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetRightMargin(0.18)
    ROOT.gPad.SetBottomMargin(0.12)
    
    h_beta_rich = ROOT.TH2F("h_beta_rich", "Velocity Smearing (RICH);True Rigidity P_{0} [GV];Measured Velocity #beta_{RICH}", 100, bins_p, 100, 0.5, 1.05)
    h_beta_rich.GetXaxis().SetTitleOffset(1.3)
    h_beta_rich.GetZaxis().SetTitle("Events / Bin")
    h_beta_rich.GetZaxis().SetTitleOffset(1.8)
    tree.Draw("beta_RICH:P0 >> h_beta_rich", "w * (beta_RICH != -999.0)", "COLZ")
    
    save_path = os.path.join(output_dir, "C2_velocity_smearing.pdf")
    c.SaveAs(save_path)
    print(f"[INFO] Saved: {save_path}")

def plot_instrumental_beta_correlation(tree, output_dir):
    """
    Plots beta_RICH vs beta_TOF to evaluate the correlation 
    between the two independent velocity measurements.
    """
    c = ROOT.TCanvas("c_beta_corr", "Beta Correlation", 800, 600)
    c.SetRightMargin(0.15)
    c.SetBottomMargin(0.12)
    
    # Linear bins focused on the ultra-relativistic regime
    h_corr = ROOT.TH2F("h_corr", "Instrumental Beta Correlation;Measured Velocity #beta_{TOF};Measured Velocity #beta_{RICH}", 150, 0.7, 1.05, 150, 0.7, 1.05)
    h_corr.Sumw2()
    
    h_corr.GetXaxis().SetTitleOffset(1.3)
    h_corr.GetZaxis().SetTitle("Events / Bin")
    h_corr.GetZaxis().SetTitleOffset(1.4)
    
    # Ensure both detectors have valid measurements
    cut = "w * (beta_TOF != -999.0 && beta_RICH != -999.0)"
    
    # Draw Y:X
    tree.Draw("beta_RICH:beta_TOF >> h_corr", cut, "COLZ")
    
    # Add ideal diagonal line (y=x)
    diag = ROOT.TLine(0.7, 0.7, 1.05, 1.05)
    diag.SetLineColor(ROOT.kRed)
    diag.SetLineStyle(2)
    diag.Draw("SAME")
    
    save_path = os.path.join(output_dir, "C3_beta_correlation.pdf")
    c.SaveAs(save_path)
    print(f"[INFO] Saved: {save_path}")

def plot_rich_resolution_curve(tree, output_dir):
    """
    Validates the RICH velocity resolution model.
    Plots the relative error (Delta beta / beta) as a function of Kinetic Energy.
    """
    c = ROOT.TCanvas("c_rich_res", "RICH Resolution", 800, 600)
    c.SetLogx()
    c.SetRightMargin(0.15) 
    c.SetLeftMargin(0.18)  # Wide left margin to fit the fractional Y-axis title
    c.SetBottomMargin(0.12)
    
    bins_ekn = create_log_bins(80, 0.1, 100.0)
    
    # Fractional notation for the Y-axis using ROOT LaTeX
    title_y = "#frac{#beta_{RICH} - #beta_{0}}{#beta_{0}}"
    
    h_res = ROOT.TH2F("h_res", f"RICH Velocity Resolution (NaF);Kinetic Energy per Nucleon E_{{kn}} [GeV/n];{title_y}", 80, bins_ekn, 100, -0.005, 0.005)
    h_res.Sumw2()
    
    # Aesthetic offsets to prevent text overlap
    h_res.GetXaxis().SetTitleOffset(1.3)
    h_res.GetYaxis().SetTitleOffset(1.8) 
    h_res.GetZaxis().SetTitle("Events / Bin")
    h_res.GetZaxis().SetTitleOffset(1.2)
    
    # Cut focused exclusively on the NaF radiator
    cut = "w * (beta_RICH != -999.0 && radiator == 1)"
    expr_res = "(beta_RICH - beta_0) / beta_0"
    
    tree.Draw(f"{expr_res}:e_kn >> h_res", cut, "COLZ")
    
    # TProfile overlaid to show the standard deviation (resolution) in each energy bin
    prof = h_res.ProfileX("prof", 1, -1, "s") 
    prof.SetMarkerColor(ROOT.kBlack)
    prof.SetLineColor(ROOT.kBlack)
    prof.SetMarkerStyle(20)
    prof.SetMarkerSize(0.6)
    prof.Draw("SAME E1")
    
    leg = ROOT.TLegend(0.22, 0.80, 0.55, 0.96)
    leg.SetBorderSize(0)
    leg.SetFillColorAlpha(ROOT.kWhite, 0.9) 
    leg.AddEntry(h_res, "Event Density", "f")
    leg.AddEntry(prof, "Standard Deviation (#sigma)", "pe")
    leg.Draw()
    
    save_path = os.path.join(output_dir, "C4_rich_resolution.pdf")
    c.SaveAs(save_path)
    print(f"[INFO] Saved: {save_path}")

def plot_rich_efficiency(tree, output_dir):
    """
    Plots the probability of a valid RICH measurement 
    as a function of the generated True Rigidity (P0).
    Utilizes TEfficiency for rigorous binomial uncertainty calculation.
    """
    c = ROOT.TCanvas("c_rich_eff", "RICH Efficiency", 800, 600)
    c.SetLogx()
    c.SetGridy() # Horizontal grid helps reading the acceptance percentage
    c.SetBottomMargin(0.12)
    c.SetLeftMargin(0.12)
    
    bins_p = create_log_bins(100, 1.0, 1000.0)
    
    # Denominator: All events that successfully passed the TOF
    h_total = ROOT.TH1F("h_total", "Total", 100, bins_p)
    h_total.Sumw2()
    
    # Numerator: Only events that produced a valid signal in the RICH
    h_passed = ROOT.TH1F("h_passed", "Passed RICH", 100, bins_p)
    h_passed.Sumw2()
    
    tree.Draw("P0 >> h_total", "w", "goff")
    tree.Draw("P0 >> h_passed", "w * (beta_RICH != -999.0)", "goff")
    
    if ROOT.TEfficiency.CheckConsistency(h_passed, h_total):
        pEff = ROOT.TEfficiency(h_passed, h_total)
        # Empty main title for publication quality
        pEff.SetTitle("RICH Detection Efficiency;True Rigidity P_{0} [GV];Efficiency #epsilon")
        pEff.SetMarkerColor(ROOT.kBlue + 2)
        pEff.SetLineColor(ROOT.kBlue + 2)
        pEff.SetMarkerStyle(20)
        pEff.SetMarkerSize(0.8)
        
        pEff.Draw("AP") # AP = Axes + Points
        
        # ROOT requires a pad update to instantiate the underlying graph of TEfficiency
        ROOT.gPad.Update()
        graph = pEff.GetPaintedGraph()
        
        # Force Y-axis limits to logical probability bounds [0, 1]
        graph.SetMinimum(0.0)
        graph.SetMaximum(1.05)
        graph.GetXaxis().SetTitleOffset(1.3)
        
        save_path = os.path.join(output_dir, "C5_rich_efficiency.pdf")
        c.SaveAs(save_path)
        print(f"[INFO] Saved: {save_path}")
    else:
        print("[ERROR] Statistical inconsistency. Numerator greater than denominator.")

# ===================================================================
# BLOCK D: MASS RECONSTRUCTION & ISOTOPIC SEPARATION
# ===================================================================

def plot_banana_curve(tree, output_dir):
    """
    The classic 'Banana Plot'.
    Plots Measured Velocity (beta) vs Measured Rigidity (P).
    """
    c = ROOT.TCanvas("c_banana", "Banana Plot", 800, 600)
    c.SetLogx()
    c.SetRightMargin(0.18)
    c.SetBottomMargin(0.12)
    
    # Focus the X-axis up to 50 GV to clearly see the curvature before beta approaches 1
    bins_p = create_log_bins(150, 1.0, 50.0) 
    
    h_banana = ROOT.TH2F("h_banana", ";Measured Rigidity P [GV];Measured Velocity #beta_{TOF}", 150, bins_p, 150, 0.5, 1.05)
    h_banana.Sumw2()
    
    # Aesthetic offsets
    h_banana.GetXaxis().SetTitleOffset(1.3)
    h_banana.GetYaxis().SetTitleOffset(1.2)
    h_banana.GetZaxis().SetTitle("Events / Bin")
    h_banana.GetZaxis().SetTitleOffset(1.4)
    
    tree.Draw("beta_TOF:P >> h_banana", "w * (beta_TOF != -999.0)", "COLZ")
    
    save_path = os.path.join(output_dir, "D1_banana_plot.pdf")
    c.SaveAs(save_path)
    print(f"[INFO] Saved: {save_path}")

def plot_mass_vs_rigidity(tree, output_dir):
    """
    Plots Reconstructed Mass vs True Rigidity (P0).
    Demonstrates the degradation of mass separation at higher energies.
    """
    c = ROOT.TCanvas("c_mass_rig", "Mass vs Rigidity", 1200, 500)
    c.Divide(2, 1)
    
    bins_p = create_log_bins(100, 1.0, 1000.0)
    bins_m = array('d', np.linspace(6.0, 15.0, 100))
    
    # --- Panel 1: Mass via TOF ---
    c.cd(1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetRightMargin(0.18)
    ROOT.gPad.SetBottomMargin(0.12)
    
    h_m_tof = ROOT.TH2F("h_m_tof", ";True Rigidity P_{0} [GV];Reconstructed Mass m_{TOF} [GeV/c^{2}]", 100, bins_p, 99, bins_m)
    h_m_tof.Sumw2()
    h_m_tof.GetXaxis().SetTitleOffset(1.3)
    h_m_tof.GetYaxis().SetTitleOffset(1.4)
    h_m_tof.GetZaxis().SetTitle("Events / Bin")
    h_m_tof.GetZaxis().SetTitleOffset(1.4)
    
    tree.Draw("m_TOF:P0 >> h_m_tof", "w * (m_TOF != -999.0)", "COLZ")  

    # --- Panel 2: Mass via RICH ---
    c.cd(2)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetRightMargin(0.18)
    ROOT.gPad.SetBottomMargin(0.12)
    
    h_m_rich = ROOT.TH2F("h_m_rich", ";True Rigidity P_{0} [GV];Reconstructed Mass m_{RICH} [GeV/c^{2}]", 100, bins_p, 99, bins_m)
    h_m_rich.Sumw2()
    h_m_rich.GetXaxis().SetTitleOffset(1.3)
    h_m_rich.GetYaxis().SetTitleOffset(1.4)
    h_m_rich.GetZaxis().SetTitle("Events / Bin")
    h_m_rich.GetZaxis().SetTitleOffset(1.4)
    
    tree.Draw("m_RICH:P0 >> h_m_rich", "w * (m_RICH != -999.0)", "COLZ")
    
    save_path = os.path.join(output_dir, "D2_mass_separation.pdf")
    c.SaveAs(save_path)
    print(f"[INFO] Saved: {save_path}")

def plot_mass_by_radiator(tree, output_dir):
    """
    Compares the reconstructed mass distributions (1D) 
    between the NaF and AGL radiators.
    """
    c = ROOT.TCanvas("c_mass_rad", "Mass by Radiator", 800, 600)
    c.SetBottomMargin(0.15)
    c.SetLeftMargin(0.15)
    
    # Independent 1D histograms (empty main title for clean LaTeX integration)
    h_naf = ROOT.TH1F("h_naf", "Reconstructed Mass by Radiator;Reconstructed Mass m_{RICH} [GeV/c^{2}];Events / Bin", 150, 4.0, 20.0)
    h_agl = ROOT.TH1F("h_agl", "Reconstructed Mass by Radiator;Reconstructed Mass m_{RICH} [GeV/c^{2}];Events / Bin", 150, 4.0, 20.0)
    h_naf.Sumw2()
    h_agl.Sumw2()
    
    # Aesthetics: Red for NaF, Blue for AGL
    h_naf.SetLineColor(ROOT.kRed + 1)
    h_naf.SetFillColorAlpha(ROOT.kRed + 1, 0.4)
    h_agl.SetLineColor(ROOT.kBlue + 1)
    h_agl.SetFillColorAlpha(ROOT.kBlue + 1, 0.4)
    
    # Cuts: Require valid mass and isolate the radiator index
    cut_naf = "w * (m_RICH != -999.0 && radiator == 1)"
    cut_agl = "w * (m_RICH != -999.0 && radiator == 2)"
    
    # Fill in background first to evaluate max heights
    tree.Draw("m_RICH >> h_agl", cut_agl, "goff")
    tree.Draw("m_RICH >> h_naf", cut_naf, "goff")
    
    # Dynamic Y-axis scaling to prevent legend overlap
    max_y = max(h_naf.GetMaximum(), h_agl.GetMaximum())
    h_agl.SetMaximum(max_y * 1.2)
    
    # Aesthetic offsets
    h_agl.GetXaxis().SetTitleOffset(1.3)
    h_agl.GetYaxis().SetTitleOffset(1.4)
    
    # Draw AGL first (usually higher statistics/broader area) and NaF on top
    h_agl.Draw("HIST")
    h_naf.Draw("HIST SAME")
    
    # Scientific Legend
    leg = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.AddEntry(h_naf, "NaF Radiator", "f")
    leg.AddEntry(h_agl, "AGL Radiator", "f")
    leg.Draw()
    
    save_path = os.path.join(output_dir, "D3_mass_by_radiator.pdf")
    c.SaveAs(save_path)
    print(f"[INFO] Saved: {save_path}")

def plot_mass_contamination(tree, output_dir):
    """
    Plots the 1D mass distributions separating the true 10B and 11B events.
    Visually justifies the need for analytical templates and Fraction Fitting.
    """
    c = ROOT.TCanvas("c_mass_cont", "Mass Contamination", 800, 600)
    c.SetBottomMargin(0.12)
    
    h_m10 = ROOT.TH1F("h_m10", ";Reconstructed Mass m_{TOF} [GeV/c^{2}];Events / Bin", 150, 4.0, 20.0)
    h_m11 = ROOT.TH1F("h_m11", ";Reconstructed Mass m_{TOF} [GeV/c^{2}];Events / Bin", 150, 4.0, 20.0)
    
    # Aesthetics
    h_m10.SetLineColor(ROOT.kBlue + 1)
    h_m10.SetFillColorAlpha(ROOT.kBlue + 1, 0.4)
    h_m11.SetLineColor(ROOT.kRed + 1)
    h_m11.SetFillColorAlpha(ROOT.kRed + 1, 0.4)
    
    # Isolate isotopes using A=10 or A=11 truth flags
    cut_10 = "w * (m_TOF != -999.0 && A == 10)"
    cut_11 = "w * (m_TOF != -999.0 && A == 11)"
    
    tree.Draw("m_TOF >> h_m10", cut_10, "goff")
    tree.Draw("m_TOF >> h_m11", cut_11, "goff")
    
    # Dynamic Y-axis scaling
    max_y = max(h_m10.GetMaximum(), h_m11.GetMaximum())
    h_m10.SetMaximum(max_y * 1.2)
    
    h_m10.GetXaxis().SetTitleOffset(1.3)
    
    h_m10.Draw("HIST")
    h_m11.Draw("HIST SAME")
    
    # Scientific Legend (using ROOT LaTeX formatting)
    leg = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.AddEntry(h_m10, "True ^{10}B", "f")
    leg.AddEntry(h_m11, "True ^{11}B", "f")
    leg.Draw()
    
    save_path = os.path.join(output_dir, "D4_mass_contamination.pdf")
    c.SaveAs(save_path)
    print(f"[INFO] Saved: {save_path}")

# ===================================================================
# EXECUTION MANAGER
# ===================================================================

if __name__ == "__main__":
    
    # 1. Initialize Environment
    print("=" * 60)
    print(" AMS-02 ADVANCED DIAGNOSTICS & PLOTTING")
    print("=" * 60)
    out_dir = setup_environment("plots_advanced")
    
    # 2. Input Data Configuration
    INPUT_FILE = "ams_b10_b11_simulation.root"
    TREE_NAME = "tree_ams"
    
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Simulation file '{INPUT_FILE}' not found.")
        print("Please run the Monte Carlo simulation first.")
        exit(1)
        
    print(f"[INFO] Opening file: {INPUT_FILE}")
    root_file = ROOT.TFile(INPUT_FILE, "READ")
    data_tree = root_file.Get(TREE_NAME)
    
    # 3. Execution Control Dictionary
    # Toggle 'True' or 'False' to run specific plots without executing the entire suite.
    PLOTS_TO_RUN = {
        # BLOCK A: Kinematics
        "kinematic_spectrum": True,
        "kinetic_energy_corr": True,
        
        # BLOCK B: Geometry
        "rich_radiator_map": True,
        "rich_spatial_density": True,
        
        # BLOCK C: Resolution
        "tracker_migration": True,
        "velocity_smearing": True,
        "beta_correlation": True,
        "rich_resolution": True,
        "rich_efficiency": True,
        
        # BLOCK D: Mass & Separation
        "banana_curve": True,
        "mass_vs_rigidity": True,
        "mass_by_radiator": True,
        "mass_contamination": True
    }
    
    # 4. Plot Dispatcher
    print("-" * 60)
    print(" Executing active plot routines...")
    print("-" * 60)
    
    if PLOTS_TO_RUN["kinematic_spectrum"]:
        print(" -> Generating Kinematic Spectrum...")
        plot_kinematic_spectrum(data_tree, out_dir)
        
    if PLOTS_TO_RUN["kinetic_energy_corr"]:
        print(" -> Generating Kinetic Energy Correlation...")
        plot_kinetic_energy_correlation(data_tree, out_dir)

    if PLOTS_TO_RUN["rich_radiator_map"]:
        print(" -> Generating RICH Radiator Map...")
        plot_rich_radiator_map(data_tree, out_dir)
        
    if PLOTS_TO_RUN["rich_spatial_density"]:
        print(" -> Generating RICH Spatial Density...")
        plot_rich_spatial_density(data_tree, out_dir)

    if PLOTS_TO_RUN["tracker_migration"]:
        print(" -> Generating Tracker Migration Matrix...")
        plot_tracker_migration(data_tree, out_dir)
        
    if PLOTS_TO_RUN["velocity_smearing"]:
        print(" -> Generating Velocity Smearing...")
        plot_velocity_smearing(data_tree, out_dir)
        
    if PLOTS_TO_RUN["beta_correlation"]:
        print(" -> Generating Beta Correlation...")
        plot_instrumental_beta_correlation(data_tree, out_dir)
        
    if PLOTS_TO_RUN["rich_resolution"]:
        print(" -> Generating RICH Resolution Curve...")
        plot_rich_resolution_curve(data_tree, out_dir)
        
    if PLOTS_TO_RUN["rich_efficiency"]:
        print(" -> Generating RICH Efficiency Curve...")
        plot_rich_efficiency(data_tree, out_dir)

    if PLOTS_TO_RUN["banana_curve"]:
        print(" -> Generating Banana Curve...")
        plot_banana_curve(data_tree, out_dir)
        
    if PLOTS_TO_RUN["mass_vs_rigidity"]:
        print(" -> Generating Mass vs Rigidity...")
        plot_mass_vs_rigidity(data_tree, out_dir)
        
    if PLOTS_TO_RUN["mass_by_radiator"]:
        print(" -> Generating Mass by Radiator...")
        plot_mass_by_radiator(data_tree, out_dir)
        
    if PLOTS_TO_RUN["mass_contamination"]:
        print(" -> Generating Mass Contamination...")
        plot_mass_contamination(data_tree, out_dir)

    # 5. Cleanup
    root_file.Close()
    print("-" * 60)
    print(" Plotting suite execution completed.")
    print("=" * 60)
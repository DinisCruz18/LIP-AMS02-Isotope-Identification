import numpy as np
import ROOT
import time
from ams_mass_templates import AMSMassTemplateAnalytic
from ams_constants import N_BINS, M_MIN, M_MAX, MASS_B10, MASS_B11, USE_TRACKER_RESOLUTION, ANALYSIS_CHANNELS

def generate_and_save_templates():
    """
    Orchestrates the generation of analytical mass templates for Boron-10 and Boron-11.
    
    This script performs the following steps:
    1. Instantiates the analytical mathematical engine.
    2. Iterates over the detector configurations and velocity bins.
    3. Converts the NumPy probability arrays into ROOT histograms (TH1D).
    4. Applies visual formatting.
    5. Exports the results: .root (for fitting) and .pdf (for reporting).
    """
    print("=" * 50)
    print(" MASS TEMPLATES EXTRACTION AND PLOTTING")
    print("=" * 50)
    
    # ===================================================================
    # 1. INITIAL CONFIGURATIONS
    # ===================================================================
    # SetBatch(True) prevents ROOT from trying to open pop-up X11 windows 
    # for each generated plot, drastically speeding up the loop.
    ROOT.gROOT.SetBatch(True) 
    
    # Mass Axis (X Axis) 
    # Broadened and with sufficient granularity to capture the smearing 
    # induced by the Tracker resolution at high energies.
    n_bins = N_BINS
    m_min = M_MIN
    m_max = M_MAX
    
    # Calculation of the bin centers for the theoretical evaluation
    delta_m = (m_max - m_min) / n_bins
    masses_x = np.linspace(m_min + delta_m/2, m_max - delta_m/2, n_bins)
    
    # ===================================================================
    # 2. ANALYTICAL ENGINE INITIALIZATION
    # ===================================================================
    template_engine = AMSMassTemplateAnalytic()
    
    # ===================================================================
    # 3. AUTOMATION LOOP (Processing and Plotting)
    # ===================================================================
    for config in ANALYSIS_CHANNELS:
        target_detector = config['detector']
        beta_bin = config['beta_bin']
        
        print(f"\n-> Processing channel: {target_detector} | #beta in {beta_bin}")
        start_time = time.time()  
        
        # String formatting for the filenames without decimal points
        b_min_str = f"{beta_bin[0]:.2f}".replace(".", "")
        b_max_str = f"{beta_bin[1]:.2f}".replace(".", "")
    
        # --- 3.1 Distribution Calculation ---
        # use_tracker_resolution=True activates the 4% relative error
        y_b10 = template_engine.compute_template(
            m_array=masses_x, m_true=MASS_B10, beta_bin=beta_bin, 
            detector=target_detector, use_tracker_resolution=USE_TRACKER_RESOLUTION
        )
        
        y_b11 = template_engine.compute_template(
            m_array=masses_x, m_true=MASS_B11, beta_bin=beta_bin, 
            detector=target_detector, use_tracker_resolution=USE_TRACKER_RESOLUTION
        )
        
        # --- 3.2 ROOT Histograms Creation ---
        hist_title = f"Mass Templates ({target_detector}, {beta_bin[0]} < #beta < {beta_bin[1]});Reconstructed Mass [GeV/c^{{2}}];Normalized Probability (dN/dm)"
        
        h_b10 = ROOT.TH1D(f"template_b10_{target_detector}", hist_title, n_bins, m_min, m_max)
        h_b11 = ROOT.TH1D(f"template_b11_{target_detector}", hist_title, n_bins, m_min, m_max)
        
        # Manual filling of the bins
        for i in range(n_bins):
            h_b10.SetBinContent(i + 1, y_b10[i])
            h_b11.SetBinContent(i + 1, y_b11[i])
            
        # --- 3.3 Styling for Presentation/Report ---
        h_b10.SetLineColor(ROOT.kBlue + 1)
        h_b10.SetLineWidth(3) 
        h_b11.SetLineColor(ROOT.kRed + 1)
        h_b11.SetLineWidth(3)
        
        # Remove irrelevant statistics box
        h_b10.SetStats(0)
        h_b11.SetStats(0)
        
        # Improve axis readability
        h_b10.GetXaxis().SetTitleSize(0.045)
        h_b10.GetYaxis().SetTitleSize(0.045)
        h_b10.GetYaxis().SetTitleOffset(1.2)

        # --- 3.4 .root File Export (For Data Analysis) ---
        root_filename = f"mass_templates_{target_detector}_beta{b_min_str}_{b_max_str}.root"
        out_file = ROOT.TFile(root_filename, "RECREATE")
        h_b10.Write("template_10B")
        h_b11.Write("template_11B")
        out_file.Close()
        
        # --- 3.5 .pdf File Export (For Visualization) ---
        canvas = ROOT.TCanvas(f"c_{target_detector}", "Templates Canvas", 800, 600)
        canvas.SetGrid()
        # Add an extra left margin so the Y-axis title isn't cut off
        canvas.SetLeftMargin(0.12)
        
        # Dynamic Y-axis adjustment to ensure the peaks do not touch the top
        max_y = max(h_b10.GetMaximum(), h_b11.GetMaximum())
        h_b10.SetMaximum(max_y * 1.25) 
        
        # Draw with "HIST" to force a continuous line, without error bars or crosses
        h_b10.Draw("HIST")
        h_b11.Draw("HIST SAME")
        
        # Legend
        legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
        legend.AddEntry(h_b10, "^{10}B Analytical", "l")
        legend.AddEntry(h_b11, "^{11}B Analytical", "l")
        legend.SetBorderSize(0) # Removes the border
        legend.SetTextSize(0.04)
        legend.Draw()
        
        pdf_filename = f"plot_templates_{target_detector}_beta{b_min_str}_{b_max_str}.pdf"
        canvas.SaveAs(pdf_filename)
        
        # ROOT Memory Cleanup
        canvas.Close()
        
        end_time = time.time()
        print(f"   -> Saved: {root_filename}")
        print(f"   -> Saved: {pdf_filename}")
        print(f"   -> Completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    generate_and_save_templates()
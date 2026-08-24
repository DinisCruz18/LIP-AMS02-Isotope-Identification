import ROOT
import sys
import ctypes
from ams_constants import N_BINS, M_MIN, M_MAX, SIMULATION_FILENAME, ANALYSIS_CHANNELS

def run_fraction_fit():
    """
    Isotopic Fraction Adjustment Engine (Template Fitting).
    
    This script orchestrates the statistical separation analysis:
    1. Imports the previously generated analytical templates.
    2. Reads the simulated data tree (Flight Data) and projects the measured masses.
    3. Uses TFractionFitter (Maximum Likelihood Estimation) to find the 
       ideal proportion of 10B and 11B that best describes the data.
    4. Generates the final publication plots and records the results.
    """
    print("=" * 60)
    print(" ISOTOPIC FRACTIONS EXTRACTION (FRACTION FITTER)")
    print("=" * 60)
    
    ROOT.gROOT.SetBatch(True) 
    ROOT.gStyle.SetOptStat(0) 
    
    # =========================================================
    # 1. ANALYSIS WINDOW CONFIGURATIONS
    # =========================================================
    
    # WARNING: Must match analyze_templates.py
    n_bins = N_BINS
    m_min = M_MIN
    m_max = M_MAX
    
    data_file = ROOT.TFile(SIMULATION_FILENAME, "READ")
    if data_file.IsZombie():
        print("ERROR: Simulation file not found.")
        sys.exit(1)
        
    data_tree = data_file.Get("tree_ams")
    
    # Structure to save the final report on the screen
    results_report = []

    # =========================================================
    # 2. FIT LOOP PER DETECTOR
    # =========================================================
    for config in ANALYSIS_CHANNELS:
        detector = config['detector']
        beta_min, beta_max = config['beta_bin']
        
        b_min_str = f"{beta_min:.2f}".replace(".", "")
        b_max_str = f"{beta_max:.2f}".replace(".", "")
        
        print(f"\n-> Starting fit for {detector} in the beta range {beta_min} - {beta_max}")
        
        templates_filename = f"mass_templates_{detector}_beta{b_min_str}_{b_max_str}.root"
        templates_file = ROOT.TFile(templates_filename, "READ")
        
        if templates_file.IsZombie():
            print(f"   [WARNING] File {templates_filename} ignored (Not found).")
            continue
            
        h_b10_theory = templates_file.Get("template_10B")
        h_b11_theory = templates_file.Get("template_11B")
        
        # TFractionFitter does not tolerate absolute bins at 0.0
        epsilon = 1e-9
        for i in range(1, h_b10_theory.GetNbinsX() + 1):
            if h_b10_theory.GetBinContent(i) <= 0.0:
                h_b10_theory.SetBinContent(i, epsilon)
            if h_b11_theory.GetBinContent(i) <= 0.0:
                h_b11_theory.SetBinContent(i, epsilon)
            
            # ROOT requires error bars on templates to initialize the Minimization
            h_b10_theory.SetBinError(i, 1e-6)
            h_b11_theory.SetBinError(i, 1e-6)

        # --- 2.1 TTree Data Projection ---
        data_title = f"Data vs Model ({detector}, {beta_min} < #beta < {beta_max});Reconstructed Mass [GeV/c^{{2}}];Events"
        h_data = ROOT.TH1D(f"h_data_{detector}", data_title, n_bins, m_min, m_max)
        h_data.Sumw2() # Poisson errors sqrt(N)
        
        # Smart selection with the pre-calculated masses in the simulation
        if detector == 'TOF':
            var_beta = "beta_TOF"
            var_mass = "m_TOF"
            detector_cut = "1" 
        elif detector == 'NaF':
            var_beta = "beta_RICH"
            var_mass = "m_RICH"
            detector_cut = "radiator == 1"
        elif detector == 'AGL':
            var_beta = "beta_RICH"
            var_mass = "m_RICH"
            detector_cut = "radiator == 2"

        # Velocity and detector restriction, and rejection of instrumental failures (-999.0)
        cut_expr = f"{var_beta} >= {beta_min} && {var_beta} < {beta_max} && {detector_cut} && {var_mass} != -999.0"
        
        data_tree.Draw(f"{var_mass} >> h_data_{detector}", f"w * ({cut_expr})", "goff")
        
        if h_data.GetEntries() < 10:
            print(f"   [WARNING] Insufficient statistics (< 10 events) for fit in {detector}.")
            continue

        # --- 2.2 Fit Engine (TFractionFitter) ---
        array_templates = ROOT.TObjArray(2)
        array_templates.Add(h_b10_theory)
        array_templates.Add(h_b11_theory)
        
        fitter = ROOT.TFractionFitter(h_data, array_templates)
        
        # The isotope cannot be "negative" nor exceed 100% of the sample
        fitter.Constrain(0, 0.0, 1.0)
        fitter.Constrain(1, 0.0, 1.0)
        
        status = fitter.Fit() 
        
        # --- 2.3 Results Extraction ---
        frac_b10 = ctypes.c_double(0.0)
        err_b10 = ctypes.c_double(0.0)
        frac_b11 = ctypes.c_double(0.0)
        err_b11 = ctypes.c_double(0.0)
        
        fitter.GetResult(0, frac_b10, err_b10)
        fitter.GetResult(1, frac_b11, err_b11)
        
        results_report.append({
            'det': detector,
            'bin': f"{beta_min}-{beta_max}",
            'f10': frac_b10.value * 100,
            'e10': err_b10.value * 100,
            'status': int(status)
        })
        
        # --- 2.4 Plotting ---
        c1 = ROOT.TCanvas(f"c_fit_{detector}", "Fraction Fit", 800, 600)
        c1.SetGrid()
        
        h_total_model = fitter.GetPlot()
        h_total_model.SetLineColor(ROOT.kRed)
        h_total_model.SetLineWidth(2)
        
        h_data.SetMarkerStyle(20)
        h_data.SetMarkerSize(1.0)
        h_data.SetLineColor(ROOT.kBlack)
        h_data.SetMarkerColor(ROOT.kBlack)
        
        # Y-axis adjustment to fit the legend
        max_y = max(h_data.GetMaximum(), h_total_model.GetMaximum())
        h_data.SetMaximum(max_y * 1.3)
        
        h_data.Draw("E P")       
        h_total_model.Draw("SAME HIST") 
        
        legend = ROOT.TLegend(0.60, 0.75, 0.88, 0.88)
        legend.AddEntry(h_data, "Simulated Data", "lep")
        legend.AddEntry(h_total_model, "Total Fit (MLE)", "l")
        legend.SetBorderSize(0)
        legend.Draw()
        
        text = ROOT.TLatex()
        text.SetNDC()
        text.SetTextSize(0.04)
        text.DrawLatex(0.15, 0.82, f"10B Fraction: {frac_b10.value*100:.1f} #pm {err_b10.value*100:.1f} %")    
        text.DrawLatex(0.15, 0.76, f"Statistics: {int(h_data.GetEntries())} events")
        
        pdf_filename = f"fit_result_{detector}_beta{b_min_str}_{b_max_str}.pdf" 
        c1.SaveAs(pdf_filename)
        
        c1.Close()
        templates_file.Close()

    # =========================================================
    # 3. FINAL REPORT
    # =========================================================
    print("\n" + "=" * 60)
    print(" FIT RESULTS SUMMARY (10B ABUNDANCE)")
    print("=" * 60)
    print(f"{'Detector':<10} | {'Velocity Window':<20} | {'10B Fraction (%)':<15} | {'Status'}")
    print("-" * 60)
    
    for res in results_report:
        status_str = "OK" if res['status'] == 0 else f"FAILED ({res['status']})"
        print(f"{res['det']:<10} | {res['bin']:<20} | {res['f10']:.2f} ± {res['e10']:.2f}    | {status_str}")
    print("=" * 60)

if __name__ == "__main__":
    run_fraction_fit()
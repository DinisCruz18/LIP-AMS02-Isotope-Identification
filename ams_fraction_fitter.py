import ROOT
import sys
import os
from ams_constants import N_BINS, M_MIN, M_MAX, SIMULATION_FILENAME, ANALYSIS_CHANNELS

def run_fraction_fit():
    """
    Isotopic Fraction Adjustment Engine (Template Fitting).
    
    This script orchestrates the statistical separation analysis:
    1. Imports the previously generated analytical templates.
    2. Reads the simulated data tree (Flight Data) and projects the measured masses.
    3. Uses RooFit (Extended Maximum Likelihood Estimation) to find the 
       ideal proportion of 10B and 11B that best describes the data.
    4. Generates the final publication plots and records the results.
    """
    print("=" * 60)
    print(" ISOTOPIC FRACTIONS EXTRACTION (ROOFIT MLE)")
    print("=" * 60)
    
    ROOT.gROOT.SetBatch(True) 
    # Silences long RooFit logs to keep the terminal clean
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)
    
    output_dir = "fits_out"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    n_bins = N_BINS
    m_min = M_MIN
    m_max = M_MAX
    
    data_file = ROOT.TFile(SIMULATION_FILENAME, "READ")
    if data_file.IsZombie():
        print("ERROR: Simulation file not found.")
        sys.exit(1)
        
    data_tree = data_file.Get("tree_ams")
    results_report = []

    for config in ANALYSIS_CHANNELS:
        detector = config['detector']
        beta_min, beta_max = config['beta_bin']
        
        b_min_str = f"{beta_min:.2f}".replace(".", "")
        b_max_str = f"{beta_max:.2f}".replace(".", "")
        
        print(f"\n-> Starting RooFit for {detector} in {beta_min} - {beta_max}")
        
        templates_filename = os.path.join("templates_out", f"mass_templates_{detector}_beta{b_min_str}_{b_max_str}.root")
        templates_file = ROOT.TFile(templates_filename, "READ")
        
        if templates_file.IsZombie():
            print(f"   [WARNING] File {templates_filename} ignored (Not found).")
            continue
            
        h_b10_theory = templates_file.Get("template_10B")
        h_b11_theory = templates_file.Get("template_11B")

        # --- 1. TTree Data Projection ---
        data_title = f"h_data_{detector}"
        h_data = ROOT.TH1D(data_title, "", n_bins, m_min, m_max)
        h_data.Sumw2() # Poisson errors sqrt(N)
        
        if detector == 'TOF':
            var_beta, var_mass, detector_cut = "beta_TOF", "m_TOF", "1" 
        elif detector == 'NaF':
            var_beta, var_mass, detector_cut = "beta_RICH", "m_RICH", "radiator == 1"
        elif detector == 'AGL':
            var_beta, var_mass, detector_cut = "beta_RICH", "m_RICH", "radiator == 2"

        cut_expr = f"{var_beta} >= {beta_min} && {var_beta} < {beta_max} && {detector_cut} && {var_mass} != -999.0"
        data_tree.Draw(f"{var_mass} >> {data_title}", f"w * ({cut_expr})", "goff")
        
        if h_data.GetEntries() < 10:
            print(f"   [WARNING] Insufficient statistics in {detector}.")
            continue

        # --- 2. RooFit Engine ---
        # Physical variable
        m_var = ROOT.RooRealVar("m", "Reconstructed Mass m [GeV/c^{2}]", m_min, m_max)
        
        # Import histograms (Weighted Data and Analytical Templates)
        roohist_data = ROOT.RooDataHist("data", "Simulated Data", ROOT.RooArgList(m_var), h_data)
        roohist_b10 = ROOT.RooDataHist("temp_b10", "Template 10B", ROOT.RooArgList(m_var), h_b10_theory)
        roohist_b11 = ROOT.RooDataHist("temp_b11", "Template 11B", ROOT.RooArgList(m_var), h_b11_theory)
        
        # Create PDFs from templates (the '0' disables interpolation, preserving the exact binning)
        pdf_b10 = ROOT.RooHistPdf("pdf_b10", "PDF 10B", ROOT.RooArgSet(m_var), roohist_b10, 0)
        pdf_b11 = ROOT.RooHistPdf("pdf_b11", "PDF 11B", ROOT.RooArgSet(m_var), roohist_b11, 0)
        
        # Free parameter: 10B Fraction (Starts at 35%)
        frac_10 = ROOT.RooRealVar("frac_10", "10B Fraction", 0.35, 0.0, 1.0)
        
        # Final Model = frac_10 * pdf_b10 + (1 - frac_10) * pdf_b11
        model = ROOT.RooAddPdf("model", "Total Model", ROOT.RooArgList(pdf_b10, pdf_b11), ROOT.RooArgList(frac_10))
        
        # SumW2Error(True) ensures the mathematical error handles Monte Carlo weights (w) correctly
        fit_result = model.fitTo(roohist_data, ROOT.RooFit.Save(True), ROOT.RooFit.SumW2Error(True), ROOT.RooFit.PrintLevel(-1))
        
        f10_val = frac_10.getVal()
        f10_err = frac_10.getError()
        
        results_report.append({
            'det': detector,
            'bin': f"{beta_min}-{beta_max}",
            'f10': f10_val * 100,
            'e10': f10_err * 100,
            'status': fit_result.status()
        })
        
        # --- 3. Plotting with RooFit ---
        c1 = ROOT.TCanvas(f"c_fit_{detector}", "RooFit", 800, 600)
        c1.SetGrid()
        c1.SetLeftMargin(0.18)
        c1.SetBottomMargin(0.12)
        
        frame = m_var.frame(ROOT.RooFit.Title(""))
        frame.SetTitle("")
        
        # Draw Data and Model
        roohist_data.plotOn(frame, ROOT.RooFit.Name("Data"), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
        model.plotOn(frame, ROOT.RooFit.Name("Model"), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.LineWidth(2))
        
        frame.GetXaxis().SetTitleOffset(1.2)
        frame.GetYaxis().SetTitleOffset(1.8)
        frame.GetYaxis().SetTitle("Events / Bin")
        frame.Draw()
        
        legend = ROOT.TLegend(0.60, 0.70, 0.88, 0.85)
        legend.AddEntry(frame.findObject("Data"), "Simulated Data", "lep")
        legend.AddEntry(frame.findObject("Model"), "Total Fit (MLE)", "l")
        legend.SetBorderSize(0)
        legend.Draw()
        
        text = ROOT.TLatex()
        text.SetNDC()
        text.SetTextSize(0.04)
        text.DrawLatex(0.24, 0.92, f"{detector} Radiator | {beta_min} < #beta < {beta_max}")
        text.DrawLatex(0.24, 0.86, f"^{{10}}B Fraction: {f10_val*100:.1f} #pm {f10_err*100:.1f} %")    
        
        pdf_filename = os.path.join(output_dir, f"fit_result_{detector}_beta{b_min_str}_{b_max_str}.pdf") 
        c1.SaveAs(pdf_filename)
        
        c1.Close()
        templates_file.Close()

    # --- 4. Final Report ---
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

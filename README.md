# LIP-AMS02-Isotope-Identification
Monte Carlo simulation and analytical mass template generation for the separation of cosmic Boron isotopes (10B/11B) using the AMS-02 detector. Developed in Python and C++ for the 2026 LIP Summer Internship program under the supervision of Prof. Fernando Barão.

## Scientific Approach & Cross-Validation
To ensure absolute scientific rigor and eliminate algorithmic bias, this project was developed by a two-person team working in parallel on two distinct codebases: **Python** and **C++**. 

This continuous cross-validation strategy—where physical results, statistical weights, and mathematical implementations were systematically compared and synchronized throughout the development—guarantees that the final isotopic separation results are robust, reproducible, and completely independent of the programming language or specific numerical libraries used.

## Python Pipeline Architecture
The Python architecture is fully modular, dividing the problem into three distinct physical and mathematical steps, all governed by a single source of truth:

* **`ams_constants.py`**: The centralized control hub. It stores all immutable physical constants, detector geometry (Tracker, TOF, RICH), empirical resolution parameterizations, and analysis hyperparameters.
* **`ams_mc_simulation.py` & `ams_root_writer.py`**: The stochastic Monte Carlo engine. It generates relativistic particle kinematics, evaluates the AMS-02 geometric acceptance, applies instrumental smearing, and safely exports the valid simulated "flight data" into a ROOT `TTree`.
* **`ams_mass_templates.py` & `analyze_templates.py`**: The deterministic analytical engine. Instead of relying on noisy histograms, it uses Bayesian probability and parallelized numerical integration (SciPy) to compute immaculate theoretical mass probability density functions (dN/dm).
* **`ams_fraction_fitter.py`**: The physics extraction module. It crosses the simulated `TTree` data with the analytical templates, utilizing ROOT's `TFractionFitter` (Maximum Likelihood Estimation) to extract the exact isotopic proportions and their statistical uncertainties.

## C++ Pipeline Architecture
*(Nota para o João: Adiciona aqui uma breve descrição da arquitetura em C++, destacando as classes principais, a integração com o ROOT e como espelha a lógica do Python para a validação cruzada.)*

* **`[File/Class Name]`**: [Brief description of what it does]
* **`[File/Class Name]`**: [Brief description of what it does]
* **`[File/Class Name]`**: [Brief description of what it does]

## Repository Structure
* `/cpp_simulation`: Contains the C++ Monte Carlo simulator source code and headers.
* `/python_analysis`: Contains the modular Python pipeline (Simulation, Templates, and Fitting modules).
* `/results`: Output directory for generated `.root` files and final `.pdf` mass spectrum plots.

## Execution Flow

### Python Environment
To reproduce the analytical results using the Python pipeline, execute the modules in the following order:
1. Configure the analysis parameters (e.g., target events, isotopic fractions, velocity bins) inside **`ams_constants.py`**.
2. Run `main.py` to generate the mock cosmic data and create the ROOT `TTree`.
3. Run `analyze_templates.py` to compute and export the theoretical probability curves.
4. Run `ams_fraction_fitter.py` to perform the Maximum Likelihood fit and extract the <sup>10</sup>B/<sup>11</sup>B abundances.

### C++ Environment
*(Nota para o João: Adiciona aqui os comandos de compilação e execução, por exemplo, `make`, `./simulate`, etc.)*
1. [Step 1: Compilation instructions]
2. [Step 2: Execution instructions]

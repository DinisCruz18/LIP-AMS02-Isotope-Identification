import time
from ams_mc_simulation import AMS02Simulator 
from ams_root_writer import AMSRootWriter
from ams_constants import TARGET_EVENTS, PROB_B10, SIMULATION_FILENAME

def main():
    """
    Execution script for the AMS-02 Monte Carlo simulation.
    
    Generates a statistical sample of cosmic Boron events (10B and 11B), 
    processes the geometric acceptance and the detector responses.
    The resulting data is exported to a 
    ROOT Tree for the subsequent construction of analytical mass templates.
    """
    print("=" * 50)
    print(" STARTING AMS-02 MONTE CARLO SIMULATION")
    print("=" * 50)
    
    # ===================================================================
    # 1. PARAMETER CONFIGURATION
    # ===================================================================
    output_filename = SIMULATION_FILENAME
    
    # Probability of 0.5 (50/50) forces equal statistics for both 
    # isotopes, which is ideal to reduce the error in template generation.
    # In a real cosmic flux analysis, ~0.3 (30% 10B) would be used.
    
    print(f"[Config] Output File: {output_filename}")
    print(f"[Config] Events to Save: {TARGET_EVENTS}")
    print(f"[Config] Boron-10 Fraction: {PROB_B10 * 100}%\n")
    
    # ===================================================================
    # 2. INFRASTRUCTURE INITIALIZATION
    # ===================================================================
    simulator = AMS02Simulator()
    writer = AMSRootWriter(filename=SIMULATION_FILENAME)    
    
    # Control variables
    saved_events = 0
    total_generated_events = 0
    start_time = time.time()
    
    # ===================================================================
    # 3. GENERATION AND RECONSTRUCTION LOOP
    # ===================================================================
    print("Starting computational processing...")
    
    while saved_events < TARGET_EVENTS:
        total_generated_events += 1
        
        # Delegates generation and smearing to the simulator class
        current_event = simulator.simulate_event()
        
        # Evaluates Geometric Acceptance: if the event has a dictionary, it passed the Trigger
        if current_event is not None:
            writer.fill_event(current_event)
            saved_events += 1
            
            # Progress feedback on the console every 10,000 events
            if saved_events % 10000 == 0:
                progress = (saved_events / TARGET_EVENTS) * 100
                print(f"   -> Progress: {saved_events} events integrated into the ROOT Tree ({progress:.0f}%)")
            
    # ===================================================================
    # 4. CLOSURE AND STATISTICS
    # ===================================================================
    print("\nFinalizing recording...")
    writer.close()
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # The acceptance rate translates the percentage of the isotropic flux incident on the
    # top of the detector that effectively crosses the active area of the lower TOF.
    acceptance_rate = (saved_events / total_generated_events) * 100
    processing_rate = total_generated_events / execution_time if execution_time > 0 else 0
    
    print("\n" + "=" * 50)
    print(" FINAL SIMULATION REPORT")
    print("=" * 50)
    print(f" Generated ROOT file  : {output_filename}")
    print(f" Total Generated (Raw): {total_generated_events}")
    print(f" Valid Events         : {saved_events}")
    print(f" Geometric Efficiency : {acceptance_rate:.2f}% (TOF Acceptance)")
    print(f" Execution Time       : {execution_time:.2f} seconds")
    print(f" Code Performance     : {processing_rate:.0f} simulations/sec")
    print("=" * 50)

if __name__ == "__main__":
    main()
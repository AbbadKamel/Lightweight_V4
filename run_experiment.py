
import os
import subprocess
import sys

def run_command(command, description):
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*80}\n")
    
    try:
        subprocess.check_call(command, shell=True)
        print(f"\nSUCCESS: {description}")
    except subprocess.CalledProcessError as e:
        print(f"\nFAILURE: {description}")
        print(f"Error: {e}")
        sys.exit(1)

def run_scenario(scenario_name):
    print(f"\n\n{'#'*80}")
    print(f"STARTING FULL PIPELINE FOR SCENARIO: {scenario_name}")
    print(f"{'#'*80}\n")
    
    # 1. Generate Master Table
    run_command(
        f"python3 Phase1/scripts/create_master_table_scenario.py {scenario_name}",
        "Phase 1: Create Master Table"
    )
    
    # 2. Prepare Training Datasets
    run_command(
        f"python3 Phase1/scripts/prepare_training_datasets_scenario.py {scenario_name}",
        "Phase 1: Prepare Training Datasets"
    )
    
    # 3. Train Models
    run_command(
        f"python3 Phase2/scripts/train_cascade_scenario.py {scenario_name}",
        "Phase 2: Train Models"
    )
    
    # 4. Calculate Thresholds
    run_command(
        f"python3 Phase3/scripts/calculate_thresholds_scenario.py {scenario_name}",
        "Phase 3: Calculate Thresholds"
    )
    
    # 5. Generate Attacks
    run_command(
        f"python3 Phase3/scripts/generate_attacks_scenario.py {scenario_name}",
        "Phase 3: Generate Attacks"
    )
    
    # 6. Evaluate
    run_command(
        f"python3 Phase3/scripts/evaluate_scenario.py {scenario_name}",
        "Phase 3: Evaluate"
    )

if __name__ == "__main__":
    # Run Scenario B (Mean Only) - Most likely to fix the noise issue
    run_scenario("B_mean_only")
    
    # Run Scenario C (Mean + Std) - To see if Std adds value or just noise
    # run_scenario("C_mean_std") 

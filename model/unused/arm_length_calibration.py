import torch
import os

def calibrate_from_dataset(data_path: str):
    """
    Loads the UIP-DB dataset and calculates the subject-specific arm lengths
    for each sequence based on the initial T-pose.
    """
    print("--- Starting Arm Length Calibration from T-Pose ---")

    # --- Configuration ---
    UWB_KEY = 'vuwb' 
    PELVIS_IDX = 5  # Sensor on the lower spine
    WRIST_IDX = 0   # Sensor on the wrist
    
    # Anthropometric ratio for splitting the total arm length
    UPPER_ARM_RATIO = 0.55
    LOWER_ARM_RATIO = 0.45
    # ---------------------

    try:
        print(f"Loading original data from: {data_path}")
        original_data = torch.load(data_path, map_location="cpu")
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"\nFATAL ERROR: The data file was not found at '{data_path}'")
        return

    try:
        uwb_data_list = original_data[UWB_KEY]
        num_sequences = len(uwb_data_list)
        print(f"Found {num_sequences} sequences to calibrate.")
    except KeyError:
        print(f"\nFATAL ERROR: The key '{UWB_KEY}' was not found in the dataset.")
        return

    calibration_results = []

    for i in range(num_sequences):
        try:
            uwb_matrix = uwb_data_list[i]
            # Get the UWB distance from the first frame (t=0), which is the T-pose
            total_length = uwb_matrix[0, PELVIS_IDX, WRIST_IDX].item()

            if torch.isnan(torch.tensor(total_length)) or total_length < 0.5:
                print(f"Warning: Invalid T-pose measurement for Sequence {i} ({total_length:.2f}m). Using fallback values.")
                # Use fallback values based on anthropometric averages if the T-pose data is bad
                l1 = 0.35 
                l2 = 0.30
            else:
                # Apply the proportional split
                l1 = total_length * UPPER_ARM_RATIO
                l2 = total_length * LOWER_ARM_RATIO

            calibration_results.append({
                "seq": i,
                "total_len": total_length,
                "L1_upper": l1,
                "L2_lower": l2
            })
            
        except IndexError:
            print(f"Warning: Could not process sequence {i}. Index out of bounds.")
            continue

    # --- Print Formatted Results Table ---
    print("\n--- Arm Length Calibration Results ---")
    print(f"{'Seq':<5} | {'T-Pose Dist (m)':<18} | {'Upper Arm L1 (m)':<18} | {'Lower Arm L2 (m)'}")
    print("-" * 65)
    for res in calibration_results:
        print(f"{res['seq']:<5} | {res['total_len']:<18.2f} | {res['L1_upper']:<18.2f} | {res['L2_lower']:.2f}")
    print("-" * 65)
    
    return calibration_results


if __name__ == "__main__":
    ORIGINAL_DATA_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"
    # Run the calibration and get the results
    calibrated_lengths = calibrate_from_dataset(ORIGINAL_DATA_PATH)
    
    # You can now use these 'calibrated_lengths' in your main IK script.
    # For example, for sequence 3:
    # l1 = calibrated_lengths[3]['L1_upper']
    # l2 = calibrated_lengths[3]['L2_lower']

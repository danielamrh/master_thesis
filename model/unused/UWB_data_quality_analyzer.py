import torch
import matplotlib.pyplot as plt
import os

def analyze_dataset(data_path: str):
    """
    Loads the UIP-DB dataset and analyzes the UWB data quality for all sequences.

    This function calculates statistics for the pelvis-to-wrist distance and
    generates a plot comparing all sequences to identify outliers.
    """
    print("--- Starting UWB Data Quality Analysis ---")

    # --- Configuration ---
    UWB_KEY = 'vuwb' 
    PELVIS_IDX = 5
    WRIST_IDX = 0
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
        print(f"Found {num_sequences} sequences to analyze.")
    except KeyError:
        print(f"\nFATAL ERROR: The key '{UWB_KEY}' was not found in the dataset.")
        return

    # --- THE FIX: Use a more compatible plot style name ---
    # Prepare for plotting
    try:
        plt.style.use('seaborn-whitegrid')
    except IOError:
        print("Warning: 'seaborn-whitegrid' style not found. Using default style.")
        # Fallback to default style if seaborn styles aren't available at all
        pass
    # ----------------------------------------------------
    
    plt.figure(figsize=(15, 8))
    
    analysis_results = []

    for i in range(num_sequences):
        try:
            uwb_matrix = uwb_data_list[i]
            distance_series = uwb_matrix[:, PELVIS_IDX, WRIST_IDX]
        except IndexError:
            print(f"Warning: Could not process sequence {i}. Index out of bounds.")
            continue

        valid_distances = distance_series[~torch.isnan(distance_series)]
        if valid_distances.numel() == 0:
            print(f"Warning: Sequence {i} contains no valid UWB data.")
            continue

        mean_dist = torch.mean(valid_distances).item()
        min_dist = torch.min(valid_distances).item()
        max_dist = torch.max(valid_distances).item()
        std_dev = torch.std(valid_distances).item()

        status = "GOOD"
        if mean_dist > 1.4 or max_dist > 2.0:
            status = "SUSPECT (Too Long)"
        elif mean_dist < 0.3:
            status = "SUSPECT (Too Short)"
        
        analysis_results.append({
            "seq": i,
            "mean": mean_dist,
            "min": min_dist,
            "max": max_dist,
            "std": std_dev,
            "status": status
        })

        plt.plot(distance_series.numpy(), label=f'Seq {i} (Mean: {mean_dist:.2f}m)', alpha=0.8)

    plt.title('UWB Distance Between Pelvis (Sensor 5) and Wrist (Sensor 0) - All Sequences', fontsize=16)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Measured Distance (m)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, 3.0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    save_path = "uwb_sequence_analysis.png"
    plt.savefig(save_path)
    print(f"\nPlot saved successfully to: {os.path.abspath(save_path)}")

    print("\n--- UWB Data Quality Analysis Results ---")
    print(f"{'Seq':<5} | {'Mean (m)':<10} | {'Max (m)':<10} | {'Min (m)':<10} | {'Std Dev':<10} | {'Status'}")
    print("-" * 65)
    for res in analysis_results:
        print(f"{res['seq']:<5} | {res['mean']:<10.2f} | {res['max']:<10.2f} | {res['min']:<10.2f} | {res['std']:<10.2f} | {res['status']}")
    print("-" * 65)


if __name__ == "__main__":
    ORIGINAL_DATA_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"
    analyze_dataset(ORIGINAL_DATA_PATH)


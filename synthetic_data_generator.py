import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.utils import shuffle

def generate_final_balanced_dataset(num_days=180, num_locations=4, positive_event_ratio=0.05):
    """
    Generates a high-quality, balanced synthetic dataset by creating distinct
    scenarios for "safe" and "failure" conditions to prevent overfitting.
    """
    print(f"Generating a high-quality balanced dataset for {num_locations} locations over {num_days} days...")
    
    total_hours = num_days * 24
    num_positive_samples = int(total_hours * num_locations * positive_event_ratio)
    num_negative_samples = int(total_hours * num_locations) - num_positive_samples

    all_records = []

    # --- 1. Generate HIGH-RISK POSITIVE Samples (Rockfall = 1) ---
    print(f"\nGenerating {num_positive_samples} high-risk 'Rockfall' samples...")
    for i in range(num_positive_samples):
        loc_id = f"Zone_{chr(65 + (i % 2))}" # Confine events to vulnerable zones A and B
        
        record = {
            'location_id': loc_id,
            'ucs_mpa': round(np.random.uniform(80, 109), 1), # Weak rock
            'seismic_zone': 'IV',
            'pga_g': round(np.random.uniform(0.25, 0.5), 2),
            'initial_water_table_m': round(np.random.uniform(15, 25), 1),
            'precipitation_rate_mm_hr': round(np.random.uniform(10, 30), 2), # High intensity rain
            'cumulative_precip_72hr': round(np.random.uniform(150, 400), 2), # High 3-day saturation
            'cumulative_precip_168hr': round(np.random.uniform(300, 700), 2),
            'ground_acceleration_g': round(np.random.uniform(0.15, 0.5), 4), # Seismic trigger
            'rockfall_event': 1
        }
        # Correlate remaining features based on triggers
        record['pore_pressure_kpa'] = record['initial_water_table_m'] * 9.81 + record['cumulative_precip_72hr'] * 2.5
        record['subsurface_displacement_mm'] = np.random.uniform(5, 20) + record['ground_acceleration_g'] * 10
        record['realtime_water_table_m'] = record['pore_pressure_kpa'] / 9.81
        all_records.append(record)

    # --- 2. Generate LOW-RISK NEGATIVE Samples (Rockfall = 0) ---
    print(f"Generating {num_negative_samples} low-risk 'No Rockfall' samples...")
    for i in range(num_negative_samples):
        loc_id = f"Zone_{chr(65 + (i % num_locations))}"

        record = {
            'location_id': loc_id,
            'ucs_mpa': round(np.random.uniform(120, 200), 1), # Strong rock
            'seismic_zone': np.random.choice(['II', 'III']),
            'pga_g': round(np.random.uniform(0.1, 0.3), 2),
            'initial_water_table_m': round(np.random.uniform(25, 40), 1),
            'precipitation_rate_mm_hr': round(np.random.uniform(0, 5), 2), # Light or no rain
            'cumulative_precip_72hr': round(np.random.uniform(0, 50), 2), # Low saturation
            'cumulative_precip_168hr': round(np.random.uniform(0, 100), 2),
            'ground_acceleration_g': round(np.random.gamma(0.5, 0.01), 4), # Background noise only
            'rockfall_event': 0
        }
        record['pore_pressure_kpa'] = record['initial_water_table_m'] * 9.81 + record['cumulative_precip_72hr'] * 1.5
        record['subsurface_displacement_mm'] = np.random.uniform(0, 2) + record['ground_acceleration_g'] * 2
        record['realtime_water_table_m'] = record['pore_pressure_kpa'] / 9.81
        all_records.append(record)
        
    # --- 3. Create, Finalize, and Shuffle the DataFrame ---
    final_df = pd.DataFrame(all_records)
    
    # Add timestamp for realism (though it will be dropped before training)
    timestamps = [datetime.now() - timedelta(minutes=i*10) for i in range(len(final_df))]
    final_df['timestamp'] = timestamps
    
    # Add surface displacement as a function of subsurface displacement
    final_df['3d_surface_coord_x'] = final_df['subsurface_displacement_mm'] * 1.2
    final_df['3d_surface_coord_y'] = final_df['subsurface_displacement_mm'] * 1.15
    final_df['3d_surface_coord_z'] = final_df['subsurface_displacement_mm'] * 1.3
    
    # Shuffle the dataset to mix positive and negative samples thoroughly
    final_df = shuffle(final_df, random_state=42).reset_index(drop=True)
    
    # Reorder columns to the desired final format
    final_cols = [
        'location_id', 'timestamp', '3d_surface_coord_x', '3d_surface_coord_y', '3d_surface_coord_z',
        'pore_pressure_kpa', 'subsurface_displacement_mm', 'precipitation_rate_mm_hr', 'ground_acceleration_g',
        'realtime_water_table_m', 'cumulative_precip_72hr', 'cumulative_precip_168hr', 'ucs_mpa', 'seismic_zone', 
        'pga_g', 'initial_water_table_m', 'rockfall_event'
    ]
    final_df = final_df[final_cols].round(4)
    
    print(f"\nSuccessfully generated and shuffled {len(final_df)} records.")
    return final_df

if __name__ == "__main__":
    OUTPUT_FILE = "final_dataset.csv"
    
    synthetic_dataset = generate_final_balanced_dataset()
    synthetic_dataset.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nDataset saved to '{OUTPUT_FILE}'")
    
    print("\n--- Summary of New Balanced Dataset ---")
    event_counts = synthetic_dataset['rockfall_event'].value_counts()
    print(event_counts)
    print(f"Percentage of rockfall events: {event_counts[1] / len(synthetic_dataset) * 100:.2f}%")

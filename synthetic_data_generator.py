import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_balanced_synthetic_data(num_days=180, num_locations=4):
    """
    Generates a more balanced synthetic dataset by making the conditions
    for a rockfall event less strict.
    """
    print(f"Generating a more balanced synthetic dataset for {num_locations} locations over {num_days} days...")

    # --- 1. Define MORE LENIENT Thresholds for Rockfall ---
    WEAK_ROCK_THRESHOLD_UCS = 110  # Increased from 100
    HIGH_SATURATION_THRESHOLD_MM = 120  # Decreased from 150
    SEISMIC_TRIGGER_THRESHOLD_G = 0.10 # Decreased from 0.15

    # --- 2. Generate Fixed Features (making 2 zones vulnerable) ---
    locations_data = []
    for i in range(num_locations):
        loc_id = f"Zone_{chr(65+i)}"
        is_vulnerable_zone = (i <= 1) # Make Zone_A and Zone_B vulnerable
        
        locations_data.append({
            'location_id': loc_id,
            'ucs_mpa': round(np.random.uniform(80, 109) if is_vulnerable_zone else np.random.uniform(120, 160), 1),
            'seismic_zone': 'IV' if is_vulnerable_zone else np.random.choice(['II', 'III']),
            'pga_g': round(np.random.uniform(0.25, 0.5) if is_vulnerable_zone else np.random.uniform(0.1, 0.3), 2),
            'initial_water_table_m': round(np.random.uniform(15, 25), 1)
        })
    df_fixed = pd.DataFrame(locations_data)
    print("\n--- Generated Fixed Features ---")
    print(df_fixed)

    # --- 3. Generate Dynamic Time-Series Data ---
    start_date = datetime.now() - timedelta(days=num_days)
    total_hours = num_days * 24
    timestamps = [start_date + timedelta(hours=i) for i in range(total_hours)]
    
    all_dynamic_data = []

    for _, location in df_fixed.iterrows():
        loc_id = location['location_id']
        subsurface_disp = 0.0
        pore_pressure = location['initial_water_table_m'] * 9.81

        for ts in timestamps:
            is_monsoon = 6 <= ts.month <= 9
            precip_rate = max(0, np.random.normal(loc=8 if is_monsoon else 1, scale=5 if is_monsoon else 2))
            
            # Increase the chance of a seismic event for more triggers
            ground_accel = max(0, np.random.gamma(0.5, 0.01))
            if np.random.random() < 0.002: # Increased from 0.001
                ground_accel = np.random.uniform(0.1, location['pga_g'])

            pore_pressure_change = (precip_rate * 1.5) - (pore_pressure * 0.005)
            pore_pressure = max(location['initial_water_table_m'] * 9.81, pore_pressure + pore_pressure_change)
            
            realtime_water_table = pore_pressure / 9.81
            subsurface_disp += 0.0005 + (pore_pressure / 8000) + (ground_accel * 0.05)

            all_dynamic_data.append({
                'location_id': loc_id, 'timestamp': ts, 'precipitation_rate_mm_hr': precip_rate,
                'pore_pressure_kpa': pore_pressure, 'subsurface_displacement_mm': subsurface_disp,
                'ground_acceleration_g': ground_accel, 'realtime_water_table_m': realtime_water_table
            })
    
    df_dynamic = pd.DataFrame(all_dynamic_data)
    df_dynamic.sort_values(by=['location_id', 'timestamp'], inplace=True)

    # --- 4. Calculate Cumulative Features and Trigger Events ---
    print("\nCalculating cumulative features and triggering rockfall events...")
    df_dynamic['cumulative_precip_72hr'] = df_dynamic.groupby('location_id')['precipitation_rate_mm_hr'].rolling(window=72, min_periods=1).sum().reset_index(0,drop=True)
    df_dynamic['cumulative_precip_168hr'] = df_dynamic.groupby('location_id')['precipitation_rate_mm_hr'].rolling(window=168, min_periods=1).sum().reset_index(0,drop=True)

    df_merged = pd.merge(df_dynamic, df_fixed, on='location_id', how='left')

    conditions = (
        (df_merged['ucs_mpa'] < WEAK_ROCK_THRESHOLD_UCS) &
        (df_merged['cumulative_precip_72hr'] > HIGH_SATURATION_THRESHOLD_MM) &
        (df_merged['ground_acceleration_g'] > SEISMIC_TRIGGER_THRESHOLD_G)
    )
    df_merged['rockfall_event'] = np.where(conditions, 1, 0)
    
    df_merged['3d_surface_coord_x'] = df_merged['subsurface_displacement_mm'] * 1.2
    df_merged['3d_surface_coord_y'] = df_merged['subsurface_displacement_mm'] * 1.15
    df_merged['3d_surface_coord_z'] = df_merged['subsurface_displacement_mm'] * 1.3
    
    final_df = df_merged.round(4)
    print(f"\nSuccessfully generated {len(final_df)} records.")
    return final_df


if __name__ == "__main__":
    OUTPUT_FILE = "final_balanced_dataset.csv"
    
    synthetic_dataset = generate_balanced_synthetic_data()
    synthetic_dataset.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nDataset saved to '{OUTPUT_FILE}'")
    
    print("\n--- Summary of Generated Rockfall Events ---")
    rockfall_events = synthetic_dataset[synthetic_dataset['rockfall_event'] == 1]
    print(f"Total rockfall events generated: {len(rockfall_events)}")
    print(f"Percentage of rockfall events: {len(rockfall_events) / len(synthetic_dataset) * 100:.2f}%")
    print(rockfall_events[['timestamp', 'location_id', 'ucs_mpa', 'cumulative_precip_72hr', 'ground_acceleration_g']].head())

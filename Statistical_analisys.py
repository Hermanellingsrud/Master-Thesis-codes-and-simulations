#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:03:35 2024

@author: hermanellingsrud
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import weibull_min
from windrose import WindroseAxes

# Function to read and preprocess CSV files
def read_and_process_csv(file_path, skip_rows=0, date_column=None):
    df = pd.read_csv(file_path, skiprows=skip_rows)
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
    return df
#%%

# Define the CSV file path
csv_file_temp = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Sørvest_F/NORA3_atm3hr_sub_lon5.0_lat56.867_20140101_20231231.csv'
csv_file_wind = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Sørvest_F/NORA3_wind_sub_lon5.0_lat56.867_20140101_20231231.csv'


# Read CSV files with necessary preprocessing
df_temp = read_and_process_csv(csv_file_temp, skip_rows=32, date_column='time')


df_wind = read_and_process_csv(csv_file_wind, skip_rows=16, date_column='time')  # Adjust if specific preprocessing is needed



#%%
# Merge the two DataFrames on 'time'
df_wind_merged = pd.merge(
    df_wind[['time', 'wind_speed_10m', 'wind_direction_10m', 'wind_speed_20m', 'wind_direction_20m',
             'wind_speed_50m', 'wind_direction_50m', 'wind_speed_100m', 'wind_direction_100m',
             'wind_speed_250m', 'wind_direction_250m']],
    df_temp[['time', 'wind_speed_150m', 'wind_direction_150m', 'wind_speed_200m', 'wind_direction_200m',
             'wind_speed_300m', 'wind_direction_300m']],
    on='time',
    how='inner'
)

# Define the output file path
output_file_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Sørvest_F/merged_wind_data_sørlig_nordsjø_2_10_years.csv'

# Save the merged DataFrame to a CSV file
df_wind_merged.to_csv(output_file_path, index=False)

# Define the heights to extract wind speeds at
heights = [10, 20, 50, 100, 150, 200, 250, 300]



#%%
# Fargekart for ulike høyder
colors = {10: 'blue', 20: 'cyan', 50: 'purple', 100: 'orange', 
          150: 'green', 200: 'yellow', 250: 'red', 300: 'brown'}



# Iterer over hver høyde og lag histogram med Weibull-tilpasning
plt.figure(figsize=(12, 14))
for i, height in enumerate(heights, 1):
    plt.subplot(4, 2, i)
    
    # Hent ut vindhastighetsdata for gjeldende høyde
    column_name = f'wind_speed_{height}m'
    if column_name not in df_wind_merged.columns:
        print(f"Kolonnen {column_name} finnes ikke i dataene.")
        continue
    
    wind_speed = df_wind_merged[column_name].dropna()
    n_data = len(wind_speed)

    # Sjekk om det er nok data
    if n_data < 10:
        print(f"For få datapunkter ({n_data}) for {height}m høyde.")
        continue

    # Print antall datapunkter
    print(f"Antall datapunkter for {height}m høyde: {n_data}")
    
    # Plot histogram av vindhastigheter
    plt.hist(wind_speed, bins=30, density=True, alpha=0.6, 
             edgecolor='black', color=colors.get(height, 'grey'))
    
    # Tilpass Weibull-fordeling til dataene
    shape, loc, scale = weibull_min.fit(wind_speed, floc=0)
    
    # Generer Weibull PDF
    x = np.linspace(min(wind_speed), max(wind_speed), 100)
    weibull_pdf = weibull_min.pdf(x, shape, loc, scale)
    plt.plot(x, weibull_pdf, 'r-', lw=2, label=f'Weibull fit\nShape: {shape:.2f}\nScale: {scale:.2f}')
    
    # Legg til tittel og etiketter
    plt.title(f'Vindhastigheter ved {height}m (Sørlige Nordsjø II)', fontsize=12)
    plt.xlabel('Vindhastighet (m/s)')
    plt.ylabel('Tetthet')
    plt.legend()
    plt.grid(True)

# Juster layout og vis plottet
plt.tight_layout()
plt.show()


#%%
from scipy.interpolate import interp1d


# Define the target heights: bottom, hub, and top
target_heights = [30, 150, 270]

# Initialiser en dictionary for å lagre interpolerte vindhastigheter
interpolated_data_sn = {height: [] for height in target_heights}

# Interpoler vindhastighetsdata for hver tidsintervall i den flettede DataFrame for Utsira
for index, row in df_wind_merged.iterrows():
    # Hent ut vindhastigheter for tilgjengelige høyder
    wind_speeds = [
        row[f'wind_speed_{h}m'] for h in heights if f'wind_speed_{h}m' in row and not pd.isna(row[f'wind_speed_{h}m'])
    ]
    valid_heights = [h for h in heights if f'wind_speed_{h}m' in row and not pd.isna(row[f'wind_speed_{h}m'])]

    # Sjekk om vi har nok data for interpolasjon
    if len(wind_speeds) < 2:
        continue

    # Interpolasjonsfunksjon
    try:
        f_interp = interp1d(valid_heights, wind_speeds, kind='linear', fill_value="extrapolate")
    except Exception as e:
        print(f"Feil ved interpolasjon på indeks {index}: {e}")
        continue

    # Interpoler vindhastigheter for målhøydene
    for height in target_heights:
        interpolated_value = f_interp(height)
        interpolated_data_sn[height].append(interpolated_value)

# Definer etiketter for høydene
height_labels = {30: 'Bottom (30m)', 150: 'Hub (150m)', 270: 'Top (270m)'}

# Lag histogrammer med Weibull-tilpasning for de interpolerte høydene
plt.figure(figsize=(15, 6))
for i, height in enumerate(target_heights, 1):
    plt.subplot(1, 3, i)
    data = interpolated_data_sn[height]

    # Sjekk om det er nok data
    if len(data) < 10:
        print(f"For få datapunkter for {height_labels[height]}.")
        plt.text(0.5, 0.5, 'Insufficient data', fontsize=14, ha='center')
        continue

    # Plot histogram av vindhastigheter
    plt.hist(data, bins=30, density=True, alpha=0.6, edgecolor='black', color='skyblue')

    # Tilpass Weibull-fordeling til dataene
    try:
        shape, loc, scale = weibull_min.fit(data, floc=0)
        # Generer Weibull PDF
        x = np.linspace(min(data), max(data), 100)
        weibull_pdf = weibull_min.pdf(x, shape, loc, scale)
        plt.plot(x, weibull_pdf, 'r-', lw=2, label=f'Weibull fit\nShape: {shape:.2f}\nScale: {scale:.2f}')
    except Exception as e:
        print(f"Feil ved Weibull-tilpasning for {height_labels[height]}: {e}")
        continue

    # Legg til etiketter og tittel
    plt.title(f'Vindhastigheter ved {height_labels[height]} (Sørlige Nordsjø II)')
    plt.xlabel('Vindhastighet (m/s)')
    plt.ylabel('Tetthet')
    plt.legend()
    plt.grid(True)

# Juster layout og vis plottene
plt.tight_layout()
plt.show()


#%%
plt.figure(figsize=(10, 6))

# Definer farger for hver høyde
colors = {30: 'blue', 150: 'green', 270: 'orange'}

# Plot histogrammer og Weibull-tilpasninger for hver målhøyde
for height in target_heights:
    data = interpolated_data_sn[height]
    
    # Sjekk om det er nok data
    if len(data) < 10:
        print(f"For få datapunkter for {height_labels[height]}.")
        continue
    
    # Plot histogram (bruker density=True for å normalisere)
    plt.hist(data, bins=30, density=True, alpha=0.4, edgecolor='black', color=colors[height],
             label=f'{height_labels[height]} Height')

    # Tilpass Weibull-fordeling
    try:
        shape, loc, scale = weibull_min.fit(data, floc=0)
        # Generer Weibull PDF
        x = np.linspace(min(data), max(data), 100)
        weibull_pdf = weibull_min.pdf(x, shape, loc, scale)
        plt.plot(x, weibull_pdf, '-', color=colors[height], lw=2, 
                 label=f'Weibull fit ({height_labels[height]})\nShape: {shape:.2f}, Scale: {scale:.2f}')
    except Exception as e:
        print(f"Feil ved Weibull-tilpasning for {height_labels[height]}: {e}")
        continue

# Legg til etiketter, tittel og legend
plt.title('Vindhastigheter ved bunn, hub og topp høyder (Sørlige Nordsjø II)')
plt.xlabel('Vindhastighet (m/s)')
plt.ylabel('Tetthet')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

#%%
# Calculate and print Weibull parameters for each target height
for height in target_heights:
    data = interpolated_data_sn[height]
    
    # Fit Weibull distribution to the data
    shape, loc, scale = weibull_min.fit(data, floc=0)
    
    # Print the parameters
    print(f"Weibull parameters for {height_labels[height]} height:")
    print(f"  Shape (k): {shape:.4f}")
    print(f"  Location (c): {loc:.4f}")
    print(f"  Scale (λ): {scale:.4f}\n")
#%%


# Ensure interpolated_data contains wind speeds for all heights
heights = [10, 20, 50, 100, 150, 200, 250, 300]
interpolated_data = {}

for height in heights:
    try:
        # Use interpolated data if available, otherwise use raw data from merged DataFrame
        if height not in interpolated_data:
            interpolated_data[height] = df_wind_merged[f'wind_speed_{height}m'].dropna().tolist()
    except KeyError:
        print(f"Error: 'wind_speed_{height}m' not found in the merged DataFrame.")

# Define wind directions for each height
wind_directions = {
    10: df_wind_merged['wind_direction_10m'],
    20: df_wind_merged['wind_direction_20m'],
    50: df_wind_merged['wind_direction_50m'],
    100: df_wind_merged['wind_direction_100m'],
    150: df_wind_merged['wind_direction_150m'],
    200: df_wind_merged['wind_direction_200m'],
    250: df_wind_merged['wind_direction_250m'],
    300: df_wind_merged['wind_direction_300m']
}




#%%
# Heights and their corresponding columns in the merged DataFrame
heights = [10, 20, 50, 100, 150, 200, 250, 300]
#colors = {10: 'blue', 20: 'cyan', 50: 'purple', 100: 'orange', 150: 'green', 200: 'yellow', 250: 'red', 300: 'brown'}

# Iterate over each height and plot the wind rose
for height in heights:
    # Extract wind speed and direction for the current height
    wind_speed = df_wind_merged[f'wind_speed_{height}m'].dropna()
    wind_direction = df_wind_merged[f'wind_direction_{height}m'].dropna()

    # Check if we have enough data to plot
    if wind_speed.empty or wind_direction.empty:
        print(f"No data available for {height}m height.")
        continue

    # Create the wind rose plot
    fig = plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax()
    ax.bar(wind_direction, wind_speed, bins=np.arange(0, 25, 5), normed=True, opening=0.8, edgecolor='white')

    # Add title and legend
    ax.set_title(f'Wind Rose at {height}m Height (Sørlige Nordsjø II)', fontsize=20)
    ax.set_legend(title="Wind Speed (m/s)", loc='lower right', fontsize= 18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Save the plot as an image file
    output_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/SNII_wind/Wind_Rose_{height}m_snii.png'
    plt.savefig(output_path, dpi = 300)
    print(f"Wind rose for {height}m saved to: {output_path}")

    # Display the plot
    plt.show()


#%%% UTSIRA

# Define the CSV file path
csv_file_temp_utsira = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Vestavind_F/NORA3_atm3hr_sub_lon4.5_lat59.2_20140101_20231231.csv'
csv_file_wind_utsira = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Vestavind_F/NORA3_wind_sub_lon4.5_lat59.2_20140101_20231231.csv'



# Read CSV files with necessary preprocessing
df_temp_utsira = read_and_process_csv(csv_file_temp_utsira, skip_rows=32, date_column='time')

df_wind_utsira = read_and_process_csv(csv_file_wind_utsira, skip_rows=16, date_column='time')  # Adjust if specific preprocessing is needed




# Merge the two DataFrames on 'time'
df_wind_merged_utsira = pd.merge(
    df_wind_utsira[['time', 'wind_speed_10m', 'wind_direction_10m', 'wind_speed_20m', 'wind_direction_20m',
             'wind_speed_50m', 'wind_direction_50m', 'wind_speed_100m', 'wind_direction_100m',
             'wind_speed_250m', 'wind_direction_250m']],
    df_temp_utsira[['time', 'wind_speed_150m', 'wind_direction_150m', 'wind_speed_200m', 'wind_direction_200m',
             'wind_speed_300m', 'wind_direction_300m']],
    on='time',
    how='inner'
)

# Verify the merged DataFrame

# Define the output file path
output_file_path_utsira = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Vestavind_F/merged_wind_data_utsira_10_years.csv'

# Save the merged DataFrame to a CSV file
df_wind_merged_utsira.to_csv(output_file_path_utsira, index=False)


#%%
# Fargekart for ulike høyder
colors = {10: 'blue', 20: 'cyan', 50: 'purple', 100: 'orange', 150: 'green', 200: 'yellow', 250: 'red', 300: 'brown'}

# Iterer over hver høyde og lag histogram med Weibull-tilpasning
plt.figure(figsize=(12, 14))
for i, height in enumerate(heights, 1):
    plt.subplot(4, 2, i)
    
    # Hent ut vindhastighetsdata for gjeldende høyde
    wind_speed = df_wind_merged_utsira[f'wind_speed_{height}m'].dropna()
    
    # Sjekk om det er nok data
    n_data = len(wind_speed)
    if wind_speed.empty:
        print(f"No data available for {height}m height.")
        continue

# Print antall datapunkter
    print(f"Number of data points for {height}m height: {n_data}")
    
    # Sjekk om det er nok data
    if wind_speed.empty:
        print(f"No data available for {height}m height.")
        continue
    
    # Plot histogram av vindhastigheter
    plt.hist(wind_speed, bins=30, density=True, alpha=0.6, edgecolor='black', color=colors[height])
    
    # Tilpass Weibull-fordeling til dataene
    shape, loc, scale = weibull_min.fit(wind_speed, floc=0)
    
    # Generer Weibull PDF
    x = np.linspace(min(wind_speed), max(wind_speed), 100)
    weibull_pdf = weibull_min.pdf(x, shape, loc, scale)
    plt.plot(x, weibull_pdf, 'r-', lw=2, label='Weibull fit')
    
    # Legg til tittel og etiketter
    plt.title(f'Wind Speeds at {height}m (Utsira Nord)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Density')
    plt.legend()

# Juster layout og vis plot
plt.tight_layout()
plt.show()

#%%
# Høyder og deres tilhørende kolonner i den flettede DataFrame for Utsira
heights = [10, 20, 50, 100, 150, 200, 250, 300]

# Iterer over hver høyde og lag vindrose
for height in heights:
    # Hent ut vindhastighet og vindretning for gjeldende høyde
    wind_speed = df_wind_merged_utsira[f'wind_speed_{height}m'].dropna()
    wind_direction = df_wind_merged_utsira[f'wind_direction_{height}m'].dropna()

    # Sjekk om vi har nok data til å plotte
    if wind_speed.empty or wind_direction.empty:
        print(f"No data available for {height}m height.")
        continue

    # Lag vindrose-plot
    fig = plt.figure(figsize=(10, 10))
    ax = WindroseAxes.from_ax()
    ax.bar(wind_direction, wind_speed, bins=np.arange(0, 25, 5), normed=True, opening=0.8, edgecolor='white')

    # Legg til tittel og legend med større tekststørrelser
    ax.set_title(f'Wind Rose at {height}m Height (Utsira Nord)', fontsize=24)
    ax.set_legend(title="Wind Speed (m/s)", loc='upper right', fontsize=20, title_fontsize=22)

    # Øk tekststørrelsen for retningene (N, NE, E, osv.)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Definer filstien for lagring av plot
    output_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Utsira_wind/Wind_Rose_{height}m_utsira.png'

    # Lagre plot som bildefil
    plt.savefig(output_path, dpi=300)
    print(f"Wind rose for {height}m saved to: {output_path}")

    # Vis plot
    plt.show()
#%%
# Initialiser en dictionary for å lagre interpolerte vindhastigheter
interpolated_data_utsira = {height: [] for height in target_heights}

# Interpoler vindhastighetsdata for hver tidsintervall i den flettede DataFrame for Utsira
for index, row in df_wind_merged_utsira.iterrows():
    # Hent ut vindhastigheter for tilgjengelige høyder
    wind_speeds = [
        row[f'wind_speed_{h}m'] for h in heights if f'wind_speed_{h}m' in row and not pd.isna(row[f'wind_speed_{h}m'])
    ]
    valid_heights = [h for h in heights if f'wind_speed_{h}m' in row and not pd.isna(row[f'wind_speed_{h}m'])]

    # Sjekk om vi har nok data for interpolasjon
    if len(wind_speeds) < 2:
        continue

    # Interpolasjonsfunksjon
    f_interp = interp1d(valid_heights, wind_speeds, kind='linear', fill_value="extrapolate")

    # Interpoler vindhastigheter for målhøydene
    for height in target_heights:
        interpolated_data_utsira[height].append(f_interp(height))

# Definer etiketter for høydene
height_labels = {30: 'Bottom (30m)', 150: 'Hub (150m)', 270: 'Top (270m)'}

# Lag histogrammer med Weibull-tilpasning for de interpolerte høydene
plt.figure(figsize=(15, 6))
for i, height in enumerate(target_heights, 1):
    plt.subplot(1, 3, i)
    data = interpolated_data_utsira[height]

    # Plot histogram av vindhastigheter
    plt.hist(data, bins=30, density=True, alpha=0.6, edgecolor='black')

    # Tilpass Weibull-fordeling til dataene
    shape, loc, scale = weibull_min.fit(data, floc=0)

    # Generer Weibull PDF
    x = np.linspace(min(data), max(data), 100)
    weibull_pdf = weibull_min.pdf(x, shape, loc, scale)
    plt.plot(x, weibull_pdf, 'r-', lw=2, label='Weibull fit')

    # Legg til etiketter og tittel
    plt.title(f'Wind Speeds at {height_labels[height]} (Utsira Nord)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()
#%%
# Kombinert histogram med Weibull-tilpasning for de tre målhøydene
plt.figure(figsize=(12, 8))

# Definer farger for hver høyde
colors = {30: 'blue', 150: 'green', 270: 'orange'}
height_labels = {30: 'Bottom (30m)', 150: 'Hub (150m)', 270: 'Top (270m)'}

# Plot histogrammer og Weibull-tilpasninger for hver høyde
for height in target_heights:
    data = interpolated_data_utsira[height]
    
    # Sjekk om det er nok data
    if len(data) == 0:
        print(f"No data available for {height}m height.")
        continue

    # Plot histogram (bruk density=True for normalisering)
    plt.hist(data, bins=30, density=True, alpha=0.4, edgecolor='black', color=colors[height],
             label=f'{height_labels[height]}')

    # Tilpass Weibull-fordeling
    shape, loc, scale = weibull_min.fit(data, floc=0)
    
    # Generer Weibull PDF
    x = np.linspace(min(data), max(data), 100)
    weibull_pdf = weibull_min.pdf(x, shape, loc, scale)
    plt.plot(x, weibull_pdf, '-', color=colors[height], lw=2, 
             label=f'Weibull fit ({height_labels[height]})')

    # Print Weibull-parametere
    print(f"Weibull parameters for {height_labels[height]}:")
    print(f"  Shape (k): {shape:.4f}")
    print(f"  Location (c): {loc:.4f}")
    print(f"  Scale (λ): {scale:.4f}\n")

# Legg til etiketter, tittel og legend
plt.title('Wind Speeds at Bottom, Hub, and Top Heights (Utsira Nord)')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.grid(True)

# Vis plot
plt.show()


#%% sammenlikning

# Sammenlign vindhastigheter ved hub-høyde (150m) mellom Sørlige Nordsjø II og Utsira Nord
plt.figure(figsize=(10, 6))

# Definer farger for de to stedene
colors = {'Sørlige Nordsjø II': 'blue', 'Utsira Nord': 'orange'}

# Hent data for hub-høyde (150m)
data_sn = interpolated_data_sn[150]  # Sørlige Nordsjø II
data_utsira = interpolated_data_utsira[150]  # Utsira Nord


# Sjekk om det er nok data
if len(data_sn) < 10:
    print("For få datapunkter for Sørlige Nordsjø II ved hub-høyde.")
if len(data_utsira) < 10:
    print("For få datapunkter for Utsira Nord ved hub-høyde.")

# Plot histogram for Southern North Sea II
plt.hist(data_sn, bins=30, density=True, alpha=0.4, edgecolor='black', color=colors['Sørlige Nordsjø II'],
         label='Sørlige Nordsjø II')

# Fit Weibull distribution for Southern North Sea II
shape_sn, loc_sn, scale_sn = weibull_min.fit(data_sn, floc=0)
x_sn = np.linspace(min(data_sn), max(data_sn), 100)
weibull_pdf_sn = weibull_min.pdf(x_sn, shape_sn, loc_sn, scale_sn)
plt.plot(x_sn, weibull_pdf_sn, '-', color=colors['Sørlige Nordsjø II'], lw=2,
         label=f'Weibull fit (Sørlige Nordsjø II)\nShape: {shape_sn:.2f}, Scale: {scale_sn:.2f}')

# Plot histogram for Utsira Nord
plt.hist(data_utsira, bins=30, density=True, alpha=0.4, edgecolor='black', color=colors['Utsira Nord'],
         label='Utsira Nord')

# Fit Weibull distribution for Utsira Nord
shape_utsira, loc_utsira, scale_utsira = weibull_min.fit(data_utsira, floc=0)
x_utsira = np.linspace(min(data_utsira), max(data_utsira), 100)
weibull_pdf_utsira = weibull_min.pdf(x_utsira, shape_utsira, loc_utsira, scale_utsira)
plt.plot(x_utsira, weibull_pdf_utsira, '-', color=colors['Utsira Nord'], lw=2,
         label=f'Weibull fit (Utsira Nord)\nShape: {shape_utsira:.2f}, Scale: {scale_utsira:.2f}')

# Add labels, title, and legend
plt.title('Comparison of Wind Speeds at Hub Height (150m)', fontsize=20)
plt.xlabel('Wind Speed (m/s)', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.tight_layout()

output_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/weibull_histogram.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Combined wind rose plot saved to: {output_path}")
# Show plot
plt.show()


#%%
# Definer cut-in og cut-out hastigheter
cut_in_speed = 3.0
cut_out_speed = 25.0
rated = 10.59

# Beregn prosentandel av nedetid for Sørlige Nordsjø II
downtime_sn = [speed for speed in data_sn if speed < cut_in_speed or speed > cut_out_speed]
downtime_percentage_sn = (len(downtime_sn) / len(data_sn)) * 100 if len(data_sn) > 0 else 0

# Beregn prosentandel av tid mellom rated og cut-out hastighet for Sørlige Nordsjø II
over_rated_sn = [speed for speed in data_sn if rated < speed <= cut_out_speed]
over_rated_percentage_sn = (len(over_rated_sn) / len(data_sn)) * 100 if len(data_sn) > 0 else 0

# Beregn prosentandel av nedetid for Utsira Nord
downtime_utsira = [speed for speed in data_utsira if speed < cut_in_speed or speed > cut_out_speed]
downtime_percentage_utsira = (len(downtime_utsira) / len(data_utsira)) * 100 if len(data_utsira) > 0 else 0

# Beregn prosentandel av tid mellom rated og cut-out hastighet for Utsira Nord
over_rated_utsira = [speed for speed in data_utsira if rated < speed <= cut_out_speed]
over_rated_percentage_utsira = (len(over_rated_utsira) / len(data_utsira)) * 100 if len(data_utsira) > 0 else 0

# Print resultatene
print("Nedetid (null produksjon) ved hub-høyde (150m):")
print(f"  Sørlige Nordsjø II: {downtime_percentage_sn:.2f}%")
print(f"  Utsira Nord: {downtime_percentage_utsira:.2f}%")
print("\nProsentandel av tid over rated hastighet men under cut-out hastighet (full eller begrenset kapasitet):")
print(f"  Sørlige Nordsjø II: {over_rated_percentage_sn:.2f}%")
print(f"  Utsira Nord: {over_rated_percentage_utsira:.2f}%")

# Beregn gjennomsnittlig vindhastighet ved hub-høyde (150m)
average_speed_sn = np.mean(data_sn)
average_speed_utsira = np.mean(data_utsira)

# Print gjennomsnittlig vindhastighet
print("\nGjennomsnittlig vindhastighet ved hub-høyde (150m):")
print(f"  Sørlige Nordsjø II: {average_speed_sn:.2f} m/s")
print(f"  Utsira Nord: {average_speed_utsira:.2f} m/s")


#%%
# Definer cut-in og cut-out hastigheter
cut_in_speed = 3.0
cut_out_speed = 25.0
rated = 10.59

# Sørlige Nordsjø II – vindkategorier
below_cut_in_sn = [speed for speed in data_sn if speed < cut_in_speed]
above_cut_out_sn = [speed for speed in data_sn if speed > cut_out_speed]
over_rated_sn = [speed for speed in data_sn if rated < speed <= cut_out_speed]

# Prosentandeler
below_cut_in_pct_sn = (len(below_cut_in_sn) / len(data_sn)) * 100
above_cut_out_pct_sn = (len(above_cut_out_sn) / len(data_sn)) * 100
over_rated_pct_sn = (len(over_rated_sn) / len(data_sn)) * 100

# Utsira Nord – vindkategorier
below_cut_in_utsira = [speed for speed in data_utsira if speed < cut_in_speed]
above_cut_out_utsira = [speed for speed in data_utsira if speed > cut_out_speed]
over_rated_utsira = [speed for speed in data_utsira if rated < speed <= cut_out_speed]

# Prosentandeler
below_cut_in_pct_utsira = (len(below_cut_in_utsira) / len(data_utsira)) * 100
above_cut_out_pct_utsira = (len(above_cut_out_utsira) / len(data_utsira)) * 100
over_rated_pct_utsira = (len(over_rated_utsira) / len(data_utsira)) * 100

# Print resultater
print("Andel av tid med vind under cut-in hastighet (< 3.0 m/s):")
print(f"  Sørlige Nordsjø II: {below_cut_in_pct_sn:.2f}%")
print(f"  Utsira Nord:        {below_cut_in_pct_utsira:.2f}%")

print("\nAndel av tid med vind over cut-out hastighet (> 25.0 m/s):")
print(f"  Sørlige Nordsjø II: {above_cut_out_pct_sn:.2f}%")
print(f"  Utsira Nord:        {above_cut_out_pct_utsira:.2f}%")

print("\nAndel av tid over rated hastighet (10.59 m/s) og under cut-out (full/flat produksjon):")
print(f"  Sørlige Nordsjø II: {over_rated_pct_sn:.2f}%")
print(f"  Utsira Nord:        {over_rated_pct_utsira:.2f}%")

# Beregn og print gjennomsnittlig vindhastighet ved hub-høyde
average_speed_sn = np.mean(data_sn)
average_speed_utsira = np.mean(data_utsira)

print("\nGjennomsnittlig vindhastighet ved hub-høyde (150m):")
print(f"  Sørlige Nordsjø II: {average_speed_sn:.2f} m/s")
print(f"  Utsira Nord:        {average_speed_utsira:.2f} m/s")


#%%


# Retrieve wind data for 150 meters height for both locations
wind_speed_sn = df_wind_merged[f'wind_speed_150m'].dropna()
wind_direction_sn = df_wind_merged[f'wind_direction_150m'].dropna()

wind_speed_un = df_wind_merged_utsira[f'wind_speed_150m'].dropna()
wind_direction_un = df_wind_merged_utsira[f'wind_direction_150m'].dropna()

# Define wind speed bins and labels for the plot (including over 20 m/s)
bins = [0, 5, 10, 15, 20, 25, 100]
bin_labels = ['0-5 m/s', '5-10 m/s', '10-15 m/s', '15-20 m/s', '20-25 m/s', '25+ m/s']

# Manually chosen colors for each wind speed bin
colors = ['#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#756bb1', '#54278f', '#4B0082']

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 12), subplot_kw=dict(projection='windrose'))

# Plot for Sørlige Nordsjø II
ax1 = axs[0]
ax1.bar(wind_direction_sn, wind_speed_sn, bins=bins, normed=True, opening=0.8, edgecolor='white', colors=colors)
ax1.set_title('Wind Rose at 150m (Sørlige Nordsjø II)', fontsize=28)
ax1.tick_params(axis='both', which='major', labelsize=22)

# Plot for Utsira Nord
ax2 = axs[1]
ax2.bar(wind_direction_un, wind_speed_un, bins=bins, normed=True, opening=0.8, edgecolor='white', colors=colors)
ax2.set_title('Wind Rose at 150m (Utsira Nord)', fontsize=28)
ax2.tick_params(axis='both', which='major', labelsize=22)

# Add a common legend at the bottom with custom labels
fig.legend(
    handles=ax1.patches[:len(bins) - 1],
    labels=bin_labels,
    title="Wind Speed (m/s)",
    fontsize=24,
    title_fontsize=26,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=6
)

# Adjust layout to make space for the legend
fig.tight_layout(rect=[0, 0.02, 1, 0.95])

# Save the figure
output_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Wind_Rose_Combined_150m_CustomColors.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Combined wind rose plot saved to: {output_path}")

# Show plot
plt.show()

#%%
from windrose import WindroseAxes
import numpy as np
import pandas as pd

def calculate_direction_percentages(wind_direction, wind_speed, bins, num_directions=16):
    # Opprett Windrose uten å vise plot
    ax = WindroseAxes.from_ax()
    ax.bar(wind_direction, wind_speed, bins=bins, normed=True, opening=0.8)
    
    # Hent tabellen fra windrose-objektet
    table = ax._info['table']
    
    # Summer prosentandelene per vindretning
    direction_percentages = table.sum(axis=0)
    
    # Beregn midtpunktene for vindretningssektorene
    sector_width = 360 / num_directions
    directions_labels = [f"{int(i * sector_width)}-{int((i + 1) * sector_width)}°" 
                         for i in range(num_directions)]
    
    # Sjekk lengder for å unngå mismatch
    assert len(direction_percentages) == len(directions_labels), "Mismatch i lengdene!"
    
    # Lag DataFrame med prosentandelene
    df_percentages = pd.DataFrame({
        'Direction': directions_labels,
        'Percentage (%)': direction_percentages
    })
    
    return df_percentages

# Sørlige Nordsjø II
df_sn = calculate_direction_percentages(wind_direction_sn, wind_speed_sn, bins)
print("Prosentandel for hver vindretning (Sørlige Nordsjø II):")
print(df_sn)

# Utsira Nord
df_un = calculate_direction_percentages(wind_direction_un, wind_speed_un, bins)
print("\nProsentandel for hver vindretning (Utsira Nord):")
print(df_un)


#%%

# Hent data for vindhastighet og tid ved hub-høyde (150m)
wind_speed_sn = df_wind_merged['wind_speed_150m'].dropna()
timestamps_sn = df_wind_merged['time'][:len(wind_speed_sn)]

wind_speed_utsira = df_wind_merged_utsira['wind_speed_150m'].dropna()
timestamps_utsira = df_wind_merged_utsira['time'][:len(wind_speed_utsira)]

# Beregn glidende gjennomsnitt (30-dagers)
rolling_sn = wind_speed_sn.rolling(window=240, center=True).mean()
rolling_utsira = wind_speed_utsira.rolling(window=240, center=True).mean()

# Opprett tidsserieplott
plt.figure(figsize=(20, 10))

# Plot for Utsira Nord
plt.plot(timestamps_utsira, wind_speed_utsira, label='Utsira Nord', color='blue', alpha=0.4)
plt.plot(timestamps_utsira, rolling_utsira, label='Utsira Nord (30-day average)', color='blue', lw=2)

# Plot for Sørlige Nordsjø II
plt.plot(timestamps_sn, wind_speed_sn, label='Sørlige Nordsjø II', color='red', alpha=0.4)
plt.plot(timestamps_sn, rolling_sn, label='Sørlige Nordsjø II (30-day average)', color='red', lw=2)


# Add labels, title, and legend
plt.title('Time Series of Wind Speed at Hub Height (150m) with Monthly Average', fontsize=24)
plt.ylabel('Wind Speed (m/s)', fontsize=20)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.grid(True)


# Vis plot
plt.show()

#%%
plt.figure(figsize=(14, 8))

# Plot 30-dagers glidende gjennomsnitt for Sørlige Nordsjø II
plt.plot(timestamps_sn, rolling_sn, label='Sørlige Nordsjø II', color='red', lw=2)

# Plot 30-dagers glidende gjennomsnitt for Utsira Nord
plt.plot(timestamps_utsira, rolling_utsira, label='Utsira Nord', color='blue', lw=2)

# Legg til etiketter, tittel og legend
plt.title('Monthly Average Wind Speed at Hub Height (150m)', fontsize=24)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Wind Speed (m/s)', fontsize=20)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()


# Vis plot
plt.show()

#%%
import seaborn as sns

timestamps_sn = pd.to_datetime(timestamps_sn)
timestamps_utsira = pd.to_datetime(timestamps_utsira)

df_sn = pd.DataFrame({'wind_speed': wind_speed_sn, 'timestamp': timestamps_sn})
df_sn['month'] = df_sn['timestamp'].dt.month

df_utsira = pd.DataFrame({'wind_speed': wind_speed_utsira, 'timestamp': timestamps_utsira})
df_utsira['month'] = df_utsira['timestamp'].dt.month

plt.figure(figsize=(15, 8))
plt.suptitle('Monthly Wind Speed Distribution at Hub Height', fontsize=24)

# Boxplot for Sørlige Nordsjø II with a red palette
plt.subplot(1, 2, 1)
sns.boxplot(x='month', y='wind_speed', data=df_sn, palette='Reds')
plt.title('Sørlige Nordsjø II', fontsize=20)
plt.xlabel('Month', fontsize=19)
plt.ylabel('Hub Wind Speed (m/s)', fontsize=16)
plt.xticks(fontsize=18)  # Increase x-tick size
plt.yticks(fontsize=18)  # Increase y-tick size



# Boksplot for Utsira Nord
plt.subplot(1, 2, 2)
sns.boxplot(x='month', y='wind_speed', data=df_utsira, palette='Blues')
plt.title('Utsira Nord', fontsize=20)
plt.xlabel('Month', fontsize=19)
plt.ylabel('Hub Wind Speed (m/s)', fontsize=16)
plt.xticks(fontsize=18)  # Increase x-tick size
plt.yticks(fontsize=18)  # Increase y-tick size

plt.tight_layout()
plt.show()

#%%
# Funksjon for å tilordne sesong basert på måned
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'

# Legg til sesonginformasjon i DataFrame for Sørlige Nordsjø II
df_sn['season'] = df_sn['timestamp'].dt.month.apply(get_season)

# Legg til sesonginformasjon i DataFrame for Utsira Nord
df_utsira['season'] = df_utsira['timestamp'].dt.month.apply(get_season)

# Lag en figur for sesongboksplot
plt.figure(figsize=(15, 8))

plt.suptitle('Seasonal Wind Speed Distribution at Hub Height', fontsize=24)


# Boksplot for Sørlige Nordsjø II
plt.subplot(1, 2, 1)
sns.boxplot(x='season', y='wind_speed', data=df_sn, palette='Reds')
plt.title('Sørlige Nordsjø II', fontsize=20)
plt.xlabel('', fontsize=1)
plt.ylabel('Hub Wind Speed (m/s)', fontsize=16)
plt.xticks(fontsize=18)  # Increase x-tick size
plt.yticks(fontsize=18)  # Increase y-tick size

# Boksplot for Utsira Nord
plt.subplot(1, 2, 2)
sns.boxplot(x='season', y='wind_speed', data=df_utsira, palette='Blues')
plt.title('Utsira Nord', fontsize=20)
plt.xlabel('', fontsize=1)
plt.ylabel('Hub Wind Speed (m/s)', fontsize=16)
plt.xticks(fontsize=18)  # Increase x-tick size
plt.yticks(fontsize=18)  # Increase y-tick size

plt.tight_layout()

output_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesong_boxplot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Combined wind rose plot saved to: {output_path}")
plt.show()

#%%
# Add wind direction data from the merged DataFrames
df_sn['wind_direction'] = df_wind_merged['wind_direction_150m'].reindex(df_sn.index)
df_utsira['wind_direction'] = df_wind_merged_utsira['wind_direction_150m'].reindex(df_utsira.index)

# Check if the columns are added successfully
print(df_sn.columns)
print(df_utsira.columns)


#%%

# Define wind speed bins and labels
bins = [0, 5, 10, 15, 20, 25, 100]
bin_labels = ['0-5 m/s', '5-10 m/s', '10-15 m/s', '15-20 m/s', '20-25 m/s', '25+ m/s']
colors = ['#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#756bb1', '#54278f', '#4B0082']

# Define seasons
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']

# Iterate over each season and create wind roses
for season in seasons:
    # Filter data for Sørlige Nordsjø II
    sn_data = df_sn[df_sn['season'] == season]
    sn_speed = sn_data['wind_speed']
    sn_direction = sn_data['wind_direction']

    # Filter data for Utsira Nord
    utsira_data = df_utsira[df_utsira['season'] == season]
    utsira_speed = utsira_data['wind_speed']
    utsira_direction = utsira_data['wind_direction']

    # Check if we have valid data for both locations
    if sn_speed.empty or sn_direction.empty or utsira_speed.empty or utsira_direction.empty:
        print(f"No data available for {season}. Skipping plot.")
        continue

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 12), subplot_kw=dict(projection='windrose'))

    # Plot for Sørlige Nordsjø II
    ax1 = axs[0]
    ax1.bar(sn_direction, sn_speed, bins=bins, normed=True, opening=0.8, edgecolor='white', colors=colors)
    ax1.set_title(f'Wind Rose (Sørlige Nordsjø II) - {season}', fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    # Plot for Utsira Nord
    ax2 = axs[1]
    ax2.bar(utsira_direction, utsira_speed, bins=bins, normed=True, opening=0.8, edgecolor='white', colors=colors)
    ax2.set_title(f'Wind Rose (Utsira Nord) - {season}', fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=18)

    # Add a common legend
    fig.legend(
        handles=ax1.patches[:len(bins) - 1],
        labels=bin_labels,
        title="Hub Wind Speed (m/s)",
        fontsize=20,
        title_fontsize=22,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=6
    )

    # Show plot
    plt.show()

#%%

## Opprett en figur med 4 rader og 2 kolonner (4 sesonger x 2 lokasjoner)
fig, axs = plt.subplots(4, 2, figsize=(12, 20), subplot_kw=dict(projection='windrose'))

# Definer sesongene og lokasjonene
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
locations = [
    ('Sørlige Nordsjø II', df_sn, axs[:, 0]),  # Sørlige Nordsjø II i første kolonne
    ('Utsira Nord', df_utsira, axs[:, 1])      # Utsira Nord i andre kolonne
]

# Iterer over lokasjoner og sesonger
for loc_name, loc_data, ax_col in locations:
    for i, season in enumerate(seasons):
        # Filtrer data for den aktuelle sesongen
        season_data = loc_data[loc_data['season'] == season]
        wind_speed = season_data['wind_speed']
        wind_direction = season_data['wind_direction']

        # Sjekk om vi har gyldig data
        if wind_speed.empty or wind_direction.empty:
            print(f"No data available for {season} at {loc_name}. Skipping plot.")
            continue

        # Velg aksen for den aktuelle sesongen
        ax = ax_col[i]
        ax.bar(
            wind_direction, wind_speed, bins=bins, normed=True,
            opening=0.8, edgecolor='white', colors=colors
        )
        ax.set_title(f'{loc_name} - {season}', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)

# Legg til en felles legend for hele figuren
fig.legend(
    handles=axs[0, 0].patches[:len(bins) - 1],
    labels=bin_labels,
    title="Hub Wind Speed (m/s)",
    fontsize=16,
    title_fontsize=18,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=6
)

# Tilpass layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.suptitle('Seasonal Wind Roses for Sørlige Nordsjø II and Utsira Nord', fontsize=24)
output_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesong_windrose.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Combined wind rose plot saved to: {output_path}")
plt.show()



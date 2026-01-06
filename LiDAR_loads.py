#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:07:43 2025

@author: hermanellingsrud
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/oblex_f1_data_dwagner/lidar/lidar_range.csv'
path_wind = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/oblex_f1_data_dwagner/lidar/lidar_ws.csv'
path_cup = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/oblex_f1_data_dwagner/mast/cup_ws_corrected_40_50_60.csv'


df_heights = pd.read_csv(path, delimiter=' ')

df_wind = pd.read_csv(path_wind, skiprows=1, delimiter=' ', index_col=0)

df_wind_cup  = pd.read_csv(path_cup, delimiter=' ',index_col=0)

df_wind_cup.index = pd.to_datetime(df_wind_cup.index)

df_wind.index = pd.to_datetime(df_wind.index)

df_wind.columns = df_heights.columns[4:]

# Konverter kolonnenavnene til numerisk format hvis det er nødvendig
df_wind.columns = df_wind.columns.astype(float)
df_wind_cup.columns = df_wind_cup.columns.astype(float)

# Definer cup-høydene (f.eks. 40, 50, 60 m) – forutsatt at disse er kolonnenavn i df_wind_cup
#cup_heights = df_wind_cup.columns
cup_height_40 = df_wind_cup.columns[0]


# Finn felles tidsindeks
common_index = df_wind.index.intersection(df_wind_cup.index)


# Oppdater df_wind med cup-data for de felles tidspunktene
df_wind.loc[common_index, cup_height_40] = df_wind_cup.loc[common_index, cup_height_40]

# Sikre at cup_height_40 er en float (dersom den ikke allerede er det)
cup_height_40 = float(df_wind_cup.columns[0])


height_30m = 30.0  # Definer 30m høyde

# Legg til 30m kolonne med samme verdier som 40m
df_wind[height_30m] = df_wind[cup_height_40]

# Reindekser kolonnene i stigende rekkefølge
df_wind = df_wind.reindex(sorted(df_wind.columns), axis=1)



# Reindekserer kolonnene slik at de blir sortert i stigende rekkefølge
df_wind = df_wind.reindex(sorted(df_wind.columns), axis=1)


times_of_interest = [
    "2015-08-13 14:20:00",
    "2015-08-13 15:40:00",
    "2015-08-13 17:20:00",
    "2015-08-13 21:40:00",
    "2015-08-14 00:40:00",
    "2015-08-14 03:40:00",
    "2015-08-14 05:20:00",
    "2015-08-14 08:20:00",
]

# Velg kun ett tidspunkt
selected_time = times_of_interest[4]  # Bruker det første tidspunktet i listen

# Hent vindprofilen for det valgte tidspunktet
profile = df_wind.loc[selected_time]

# Sørg for at høydene er i numerisk form
heights = profile.index.astype(float)
windspeeds = profile.values

# Opprett en figur
plt.figure(figsize=(6, 8))

# Plot vindhastighet (x) mot høyde (y)
plt.plot(windspeeds, heights, label=str(selected_time), linewidth=2)

# Tilpass akser, tittel og legg til legend
plt.xlabel("Wind speed (m/s)", fontsize=18)
plt.ylabel("Height (m)", fontsize=18)
plt.title(f"LIDAR Wind profile {selected_time}", fontsize=18)
plt.grid(True)

# Øk tick-størrelse for både x- og y-aksen
plt.tick_params(axis='both', which='major', labelsize=14)

# Øk legend-størrelsen

plt.tight_layout()
plt.ylim(0, 300)
plt.xlim(5,18)

plt.show()

heights = df_wind.columns.to_numpy()
wind_speeds = df_wind.loc[selected_time].to_numpy()


# Filtrer ut NaN-verdier
valid_mask = ~np.isnan(wind_speeds)  # Lager en maske som er True for gyldige verdier
valid_heights = heights[valid_mask]  # Beholder kun gyldige høyder
valid_wind_speeds = wind_speeds[valid_mask]  # Beholder kun gyldige vindhastigheter


#%% airfoil properties
# Define the file path
file_path_airfoil_properties = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/IEA-15-240-RWT_AeroDyn15_blade.dat'

# Initialize lists to store extracted data
blade_span = []
blade_curve_ac = []
blade_sweep_ac = []
blade_curve_angle = []
blade_twist = []
blade_chord = []

# Open the file and read the data
with open(file_path_airfoil_properties, 'r') as file:
    # Skip the header lines
    for _ in range(6):  # Adjust the number based on how many header lines to skip
        file.readline()
    
    # Read and extract data
    for line_number, line in enumerate(file, start=7):  # Start at line 7 (after skipping headers)
        # Debug: print the current line being processed
        #print(f"Processing line {line_number}: {line.strip()}")
        
        # Split the line into components
        values = line.split()
        if len(values) >= 6:  # Ensure there are enough values in the line
            # Append the values to the corresponding lists
            try:
                blade_span.append(float(values[0]))
                blade_curve_ac.append(float(values[1]))
                blade_sweep_ac.append(float(values[2]))
                blade_curve_angle.append(float(values[3]))
                blade_twist.append(float(values[4]))
                blade_chord.append(float(values[5]))
            except ValueError as e:
                print(f"Error converting values on line {line_number}: {e}")
        else:
            print(f"Insufficient values on line {line_number}: {values}")

#%% rotor preformance

preformance_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Alle_verdier/IEA-15-240-RWT_tabular/Rotor Performance-Table 1.csv'

preformance_data = pd.read_csv(preformance_path, sep=';', decimal=',')

wind_speeds_preformance = preformance_data['Wind [m/s]']
pitch_angles = preformance_data['Pitch [deg]']
rotor_speeds = preformance_data['Rotor Speed [rpm]']
ct_values = preformance_data['Thrust Coefficient [-]']
cp_values = preformance_data['Aero Power Coefficient [-]']
power = preformance_data['Power [MW]']

# Function to calculate axial induction factor from Ct
def calculate_axial_induction_factor(Ct):
    return (1 - np.sqrt(1 - Ct)) / 2

# Calculate axial induction factors for each Ct value
a_values = calculate_axial_induction_factor(ct_values)


# Function to find or interpolate `a` based on wind speed
def get_axial_induction_factor(wind_speed, wind_speeds_performance, ct_values):
    """
    Interpolates Ct based on wind speed and calculates the axial induction factor `a`.
    
    Args:
        wind_speed (float): The current wind speed at the section.
        wind_speeds_performance (array): Array of wind speeds for which Ct values are provided.
        ct_values (array): Array of Ct values corresponding to the wind speeds.

    Returns:
        float: Calculated axial induction factor `a` for the given wind speed.
    """
    # Interpolate `Ct` based on the actual wind speed at the section
    ct_interpolated = np.interp(wind_speed, wind_speeds_performance, ct_values)
    # Calculate axial induction factor `a` from the interpolated `Ct`
    return (1 - np.sqrt(1 - ct_interpolated)) / 2

pitch_angles[0] = pitch_angles[0] +0.7
pitch_angles[1] = pitch_angles[1] +0.9
pitch_angles[2] = pitch_angles[2] +0.9
pitch_angles[3] = pitch_angles[3] +0.9
pitch_angles[4] = pitch_angles[4] +1
pitch_angles[5] = pitch_angles[5] +1
pitch_angles[6] = pitch_angles[6] +1.4
pitch_angles[7] = pitch_angles[7] +1.6
pitch_angles[8] = pitch_angles[8] +1.6
pitch_angles[9] = pitch_angles[9] +1.6
pitch_angles[10] = pitch_angles[10] +1.9

pitch_angles[28] = pitch_angles[28] +1.69
pitch_angles[29] = pitch_angles[29] +1.63
pitch_angles[30] = pitch_angles[30] +0.84
pitch_angles[31] = pitch_angles[31] +0.7
pitch_angles[32] = pitch_angles[32] +0.68
pitch_angles[33] = pitch_angles[33] +0.63
pitch_angles[34] = pitch_angles[34] +0.63
pitch_angles[35] = pitch_angles[35] +0.67
pitch_angles[36] = pitch_angles[36] +0.66
pitch_angles[37] = pitch_angles[37] +0.7
pitch_angles[38] = pitch_angles[38] +0.75
pitch_angles[39] = pitch_angles[39] +0.8
pitch_angles[40] = pitch_angles[40] +0.87
pitch_angles[41] = pitch_angles[41] +0.88
pitch_angles[42] = pitch_angles[42] +1
pitch_angles[43] = pitch_angles[43] +1# 20m/s
pitch_angles[44] = pitch_angles[44] +1.1
pitch_angles[45] = pitch_angles[45] +1.15# 21
pitch_angles[46] = pitch_angles[46] +1.22
pitch_angles[47] = pitch_angles[47] +1.28
pitch_angles[48] = pitch_angles[48] +1.38
pitch_angles[49] = pitch_angles[49] +1.43

#%% General Constants and Physical Parameters

rho = 1.225  # Air density in kg/m^3
hub_height = 150  # Height of the rotor hub (in meters)
rotor_diameter = 240  # Rotor diameter (in meters)
rotor_radius = rotor_diameter / 2  # Rotor radius (half of the diameter, in meters)

# Blade tip and bottom heights based on rotor diameter and hub height
H_top = hub_height + rotor_radius  # Height of the blade tip when at the top (270 m)
H_bottom = hub_height - rotor_radius  # Height of the blade tip when at the bottom (30 m)

# Wind Speed Parameters

cut_in_speed = 3.0  # Cut-in wind speed (in m/s)
rated_wind = 10.59  # Rated wind speed (in m/s)
cut_out_speed = 25.0  # Cut-out wind speed (in m/s)

# Rotor Speed Parameters

min_rot = 5.0  # Minimum rotational speed (in RPM)
rated_speed_rot = 7.56  # Rated rotational speed (in RPM), at rated wind speed (10.59 m/s and above)

# Induction Factors

a_prime = 0  # Initial guess for the tangential induction factor (a')

# Blade Geometry and Position

# Define blade positions (angles from 0° to 360° around the rotor)
blade_positions = np.linspace(0, 360, num=360, endpoint=False)

# Heights of the blade sections during rotation, based on cosine of the angles
blade_heights = hub_height + rotor_radius * np.cos(np.radians(blade_positions))

# Wind Speed Interpolation

# Interpolate wind speeds at these blade heights based on wind data from different heights
U = np.interp(blade_heights, heights, wind_speeds)  # Wind speed at each blade position (interpolated)

# Wind speed at the hub height (also interpolated)
wind_speed_hub = np.interp(hub_height, heights, wind_speeds)

# Interpolate wind speeds at H_top and H_bottom
wind_speed_top = np.interp(H_top, heights, wind_speeds)
wind_speed_bottom = np.interp(H_bottom, heights, wind_speeds)

# Calculate the shear exponent (alpha) using the power law formula
shear_exponent = np.log(wind_speed_top / wind_speed_bottom) / np.log(H_top / H_bottom)

# Definer lagringsbanen
save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/LIDAR/Diskusjon"

# Sørg for at mappen eksisterer
os.makedirs(save_path, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(blade_positions, U, label='Wind Speed at Blade Position', color='red', linestyle='-', markersize=3)
plt.xlabel('Blade Position (degrees)', fontsize=14)
plt.ylabel('Wind Speed (m/s)', fontsize=14)
plt.title('Wind Speed Experienced During Rotation', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
#plt.legend(fontsize=12)
plt.xticks(np.arange(0, 361, 45), fontsize=12)  # Show every 45 degrees for better readability
plt.yticks(fontsize=12)

# Definer filnavnet
filename = "Wind_Speed_Rotation.png"
full_save_path = os.path.join(save_path, filename)

# Lagre figuren
plt.savefig(full_save_path, dpi=300, bbox_inches='tight')

# Save the figure
# rotation_plot_path = f"/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/LLJ_eksempler/wind_rotation_{specific_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"
# plt.savefig(rotation_plot_path, dpi=300)
plt.show()


# %% Rotational Speed Calculation

# Interpolate rotor speeds based on wind speed, setting to 0 if below 3 m/s or above 25 m/s
rot_speed_hub = np.where(
    (wind_speed_hub < 3) | (wind_speed_hub > 25), 
    0, 
    np.interp(wind_speed_hub, wind_speeds_preformance, rotor_speeds)
)

# Convert hub height rotor speed from RPM to rad/s
omega_hub = (2 * np.pi * rot_speed_hub) / 60


#%% Air foils

# Define the folder path where your files are stored
folder_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/'

# Initialize a dictionary to hold all the aerodynamic coefficients for all airfoils
aero_data_all = {}

# Function to extract aerodynamic coefficients from a file
def extract_aero_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize the dictionary to store aoa, cl, cd, and cm
    aero_data = {'aoa': [], 'cl': [], 'cd': [], 'cm': []}

    # Find the line with the number of angles
    for line in lines:
        if 'NumAlf' in line:
            num_aoa = int(line.split()[0])  # Get number of angles
            break

    # Read the aerodynamic coefficients
    for line in lines[lines.index(line) + 1:lines.index(line) + 1 + num_aoa]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                # Attempt to convert each part to a float
                aoa_value = float(parts[0])
                cl_value = float(parts[1])
                cd_value = float(parts[2])
                cm_value = float(parts[3])
                
                # Append values to their respective lists
                aero_data['aoa'].append(aoa_value)
                aero_data['cl'].append(cl_value)
                aero_data['cd'].append(cd_value)
                aero_data['cm'].append(cm_value)
            except ValueError:
                # Skip lines that contain invalid float values
                continue
    
    # Convert lists to numpy arrays for easier manipulation
    aero_data['aoa'] = np.array(aero_data['aoa'])
    aero_data['cl'] = np.array(aero_data['cl'])
    aero_data['cd'] = np.array(aero_data['cd'])
    aero_data['cm'] = np.array(aero_data['cm'])

    return aero_data

# Extract data for all 50 airfoils
for i in range(50):
    file_name = f'IEA-15-240-RWT_AeroDyn15_Polar_{i:02d}.dat'
    file_path = os.path.join(folder_path, file_name)
    
    # Extract data and store it in a dictionary
    aero_data_all[f'airfoil_{i:02d}'] = extract_aero_data(file_path)


# Initialize lists to store data for all airfoils
airfoil_key = []
polar_data = []
aoa = []
cl = []
cd = []
cm = []


# Loop through all airfoils and process the data
for i in range(50):
    # Generate airfoil key and store it in the list
    airfoil_key.append(f'airfoil_{i:02d}')
    
    # Retrieve polar data and store it in the list
    polar_data.append(aero_data_all[airfoil_key[i]])

    # Access aoa, cl, cd, cm for each airfoil and store them in corresponding lists
    aoa.append(polar_data[i]['aoa'])
    cl.append(polar_data[i]['cl'])
    cd.append(polar_data[i]['cd'])
    cm.append(polar_data[i]['cm'])



#%% Wind for each section
# Extend blade span to include the rotor radius (120 meters)
blade_span_total = np.append(blade_span, 120)  # Now extends to the full rotor radius (120 meters)
num_blade_sections = len(blade_span_total)

# Blade heights are based on the span points (measured from the hub)
blade_heights_sections = np.array(blade_span_total)

# Define blade positions for a full rotation (0° to 360°)
blade_positions = np.linspace(0, 360, 360)  # 100 points around the rotor (can increase to 360 for higher resolution)

# Initialize arrays for storing the blade section heights and wind speeds during rotation
blade_section_rotation = np.zeros((num_blade_sections, len(blade_positions)))  # Height of each section during rotation
wind_speed_rotation = np.zeros((num_blade_sections, len(blade_positions)))    # Wind speed at each section during rotation

# Loop over each blade section and calculate the height during a full 360-degree rotation
for section in range(num_blade_sections):
    # Calculate the height of each section during the rotation (cosine variation)
    blade_section_rotation[section, :] = hub_height + blade_heights_sections[section] * np.cos(np.radians(blade_positions))

    # Interpolate the wind speed at each section height during the rotation
    wind_speed_rotation[section, :] = np.interp(blade_section_rotation[section, :], heights, wind_speeds)


plt.figure(figsize=(10,5))  # Set figure size

plt.plot(blade_positions, wind_speed_rotation[-1,:], label='Tip of the blade', linestyle='-')
plt.plot(blade_positions, wind_speed_rotation[1,:], label='Close to the rotor', linestyle='--')

plt.xlabel("Blade Position [degrees]", fontsize = 18)
plt.ylabel("Wind Speed [m/s]",fontsize = 18)
#plt.title("Wind Speed variations at different blade sections")
plt.legend(fontsize = 15)
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid with dashed lines for clarity
plt.xticks(fontsize=16)  # Increase tick size
plt.yticks(fontsize=16) 

save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/wind_speed_variation.png"

#plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save figure with high resolution

plt.show()
#%% relative velocity

blade_span_total_50 = blade_span_total[:50]  # Keep only the first 50 sections
num_blade_sections_50 = len(blade_span_total_50)
        
# Initialize arrays for lambda, b, phi values, and other parameters for each section
phi_values_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
w_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
rot_speed_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
omega_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
u_sections = np.zeros((num_blade_sections_50, len(blade_positions)))  # Effective wind speed with axial induction
v_app_sections = np.zeros((num_blade_sections_50, len(blade_positions)))  # Apparent wind speed
a_sections = np.zeros((num_blade_sections_50, len(blade_positions)))  # Axial induction factors



# Loop over each blade section and calculate the height during a full 360-degree rotation
for section in range(num_blade_sections_50):  # Use len(blade_span_total_50) for range
    # Calculate r_central for each section (average radius of the section)
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section

    # Calculate the effective wind speed with axial induction for each rotation angle
    for i in range(len(blade_positions)):  # Ensure you're iterating over indices
        # Get the wind speed at this section and rotation angle from `wind_speed_rotation`
        wind_speed_current = wind_speed_rotation[section, i]

        # Dynamically determine `a` based on the wind speed at the current position
        a_section = get_axial_induction_factor(wind_speed_hub, wind_speeds_preformance, ct_values)
        a_sections[section, i] = a_section  # Store `a` for later use or analysis
        
    

        # Apply axial induction to the wind speed for the current section
        u_sections[section, i] = wind_speed_current * (1 - a_section)

        # Use the hub rotor speed for all positions
        rot_speed_sections[section, i] = rot_speed_hub

        # Convert the rotor speed from RPM to rad/s
        omega_sections[section, i] = omega_hub

        # Calculate tangential velocity with tangential induction factor (w = omega * r * (1 + a_prime))
        w_sections[section, i] = omega_sections[section, i] * r_central * (1 + a_prime)

        # Calculate inflow angle (phi)
        phi_values_sections[section, i] = np.arctan(u_sections[section, i] / w_sections[section, i])

        # Calculate apparent wind speed
        v_app_sections[section, i] = np.sqrt(u_sections[section, i]**2 + w_sections[section, i]**2)


# Calculate the tip speed using the hub angular velocity and the rotor radius
v_tip = omega_hub * rotor_radius


# Calculate the Tip Speed Ratio (TSR)
tsr = v_tip / wind_speed_hub

print(tsr)

#%%
# Calculate the mean initial wind speed for each section (before axial induction)
mean_initial_wind_speed = np.mean(wind_speed_rotation, axis=1)

# Calculate the minimum and maximum initial wind speeds for each section
min_initial_wind_speed = np.min(wind_speed_rotation, axis=1)
max_initial_wind_speed = np.max(wind_speed_rotation, axis=1)

# Define save path in the correct folder
save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/mean_initial_wind_speed.png"

# Create figure
plt.figure(figsize=(10, 5))

# Plot mean wind speed with markers
plt.plot(blade_span_total, mean_initial_wind_speed, marker='o', linestyle='-', color='blue', label='Mean Wind Speed')

# Fill between min and max range
plt.fill_between(blade_span_total, min_initial_wind_speed, max_initial_wind_speed, color='blue', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel('Blade Span (m)', fontsize=18)
plt.ylabel('Wind Speed (m/s)', fontsize=18)
plt.title('Wind Speed Along the Blade Span', fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='lower left', frameon=True)

# Save the figure with high resolution in the correct directory
#plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Show plot
plt.show()


#%%
# Calculate the mean axial induction factor for each section along the blade
mean_axial_induction_factors = np.mean(a_sections, axis=1)


# Calculate the minimum and maximum axial induction factors for each section
min_axial_induction_factors = np.min(a_sections, axis=1)
max_axial_induction_factors = np.max(a_sections, axis=1)

# Plotting the mean axial induction factor along the blade span with range shading
plt.figure(figsize=(10, 6))
plt.plot(blade_span_total_50, mean_axial_induction_factors, marker='o', linestyle='-', color='b', label='Mean Axial Induction Factor')
plt.fill_between(blade_span_total_50, min_axial_induction_factors, max_axial_induction_factors, color='blue', alpha=0.2, label='Min-Max Range')

# Set labels and title
plt.xlabel('Blade Span (m)')
plt.ylabel('Axial Induction Factor (a)')
plt.title('Mean Axial Induction Factor Along the Blade Span')
plt.grid(True)
plt.ylim(0, 0.5)
# Place the legend outside of the plot

plt.legend(loc='upper left')

#plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/22.10.figures/mean_axial_induction_factor.png', dpi=300, bbox_inches='tight')


# Show the plot
plt.show()

        
#%%

# Calculate the mean effective wind speed after axial induction for each section
mean_effective_wind_speed = np.mean(u_sections, axis=1)

# Calculate the minimum and maximum effective wind speeds for each section
min_effective_wind_speed = np.min(u_sections, axis=1)
max_effective_wind_speed = np.max(u_sections, axis=1)

# Plot the mean effective wind speed along the blade span
plt.figure(figsize=(10, 6))
plt.plot(blade_span_total_50, mean_effective_wind_speed, marker='o', linestyle='-', color='g', label='Mean Effective Wind Speed')

# Use fill_between to show the range (min to max) as a shaded area
plt.fill_between(blade_span_total_50, min_effective_wind_speed, max_effective_wind_speed, color='green', alpha=0.2, label='Min-Max Range')

# Set labels and title
plt.xlabel('Blade Span (m)')
plt.ylabel('Effective Wind Speed (m/s)')
plt.title('Mean Effective Wind Speed After Axial Induction Along the Blade Span')

# Show grid
plt.grid()
# Place the legend outside of the plot
plt.legend(loc='upper left')

#plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/22.10.figures/mean_effective_wind_speed.png', dpi=300, bbox_inches='tight')


# Show plot
plt.show()



#%%
mean_v_app = np.mean(v_app_sections,axis=1)

# Calculate the minimum and maximum apparent wind speeds for each section
min_v_app = np.min(v_app_sections, axis=1)
max_v_app = np.max(v_app_sections, axis=1)

# Define correct save path
save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/mean_apparent_wind_speed.png"

# Create figure
plt.figure(figsize=(10, 5))

# Plot mean apparent wind speed
plt.plot(blade_span_total_50, mean_v_app, marker='o', linestyle='-', color='b', label='Mean Apparent Wind Speed')
plt.fill_between(blade_span_total_50, min_v_app, max_v_app, color='blue', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel('Blade Span (m)', fontsize=18)
plt.ylabel('Apparent Wind Speed (m/s)', fontsize=18)
plt.title('Apparent Wind Speed Along the Blade Span', fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='upper left', frameon=True)

# Save figure in the correct directory with high resolution
#plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Show plot
plt.show()
#%% blade pitch 
# Initialize a 2D array for blade pitch values at each section during the full rotation
blade_pitch_values_sections = np.zeros((num_blade_sections_50, len(blade_positions)))

# Loop through each blade section
for section in range(num_blade_sections_50):
    # Loop through each blade position to interpolate pitch values based on wind speed
    for i, blade_position in enumerate(blade_positions):

        # Interpolate the blade pitch using the wind speed and performance data
        blade_pitch_values_sections[section, i] = np.interp(wind_speed_hub, wind_speeds_preformance, pitch_angles)


#%%
# Convert phi_values_sections to degrees
phi_values_sections_degrees = np.rad2deg(phi_values_sections)

# Calculate the average, minimum, and maximum of phi_values_sections in degrees
average_phi_sections = np.mean(phi_values_sections_degrees, axis=1)
min_phi_sections = np.min(phi_values_sections_degrees, axis=1)
max_phi_sections = np.max(phi_values_sections_degrees, axis=1)

# Define the correct save path
save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/average_phi_values_degrees.png"

# Create figure
plt.figure(figsize=(10, 5))

# Plot average inflow angle
plt.plot(blade_span_total_50, average_phi_sections, 'b-o', label="Average Inflow Angle")

# Fill between min and max range
plt.fill_between(blade_span_total_50, min_phi_sections, max_phi_sections, color='blue', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel("Blade Span (m)", fontsize=18)
plt.ylabel("Inflow Angles (Degrees)", fontsize=18)
plt.title("Inflow Angles Along the Blade Span", fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='upper right', frameon=True)

# Save the figure with high resolution in the correct directory
#plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Show plot
plt.show()



#%% Angle of Attack (Alpha) Calculation
# Initialize a 2D array for angle of attack (alpha) for each section during the rotation
alpha_sections = np.zeros_like(phi_values_sections)

# Loop through each section to calculate the angle of attack (alpha)
for section in range(num_blade_sections_50):
    # Convert inflow angle (phi) from radians to degrees for the current section
    phi_deg_sections = np.degrees(phi_values_sections[section, :])
    
    # Calculate the angle of attack: alpha = phi (deg) - blade pitch (deg) - blade twist (deg)
    alpha_sections[section, :] = phi_deg_sections - blade_pitch_values_sections[section, :] - blade_twist[section]

# Calculate the average angle of attack (alpha) along the blade span
average_alpha_sections = np.mean(alpha_sections, axis=1)

# Calculate the minimum and maximum angle of attack for each section
min_alpha_sections = np.min(alpha_sections, axis=1)
max_alpha_sections = np.max(alpha_sections, axis=1)

# Define the correct save path
save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/average_angle_of_attack.png"

# Create figure
plt.figure(figsize=(10, 5))

# Plot average angle of attack
plt.plot(blade_span_total_50, average_alpha_sections, 'g-o', label="Average Angle of Attack (α)")

# Fill between min and max range
plt.fill_between(blade_span_total_50, min_alpha_sections, max_alpha_sections, color='green', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel("Blade Span (m)", fontsize=18)
plt.ylabel("Angle of Attack (Degrees)", fontsize=18)
plt.title("Angle of Attack (α) Along the Blade Span", fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='upper right', frameon=True)

# Save the figure with high resolution in the correct directory
#plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Show plot
plt.show()

#%% C_l and C_d
# Initialize arrays for Cl and Cd for each section
Cl_sections = np.zeros_like(alpha_sections)
Cd_sections = np.zeros_like(alpha_sections)

# Loop over each blade section and assign the appropriate airfoil data
for section in range(num_blade_sections_50):
    # Ensure the airfoil index does not exceed the available airfoils (max index = 49)
    #airfoil_index = min(section, 49)  # Use airfoil 49 for sections beyond the 50th airfoil

    # Get the corresponding airfoil's data (Cl, Cd)
    Cl_current = cl[section]  # Lift coefficient values (198 points)
    Cd_current = cd[section]  # Drag coefficient values (198 points)
    aoa_current = aoa[section]  # Angle of attack values (198 points)

    # Clip the angle of attack values to fit within the range of aoa_current
    alpha_clipped = np.clip(alpha_sections[section, :], aoa_current.min(), aoa_current.max())

    # Interpolate Cl and Cd based on the clipped angle of attack (alpha) for this section
    Cl_sections[section, :] = np.interp(alpha_clipped, aoa_current, Cl_current)
    Cd_sections[section, :] = np.interp(alpha_clipped, aoa_current, Cd_current)
    
# Calculate the average Cl and Cd along the blade
average_Cl_sections = np.mean(Cl_sections, axis=1)
average_Cd_sections = np.mean(Cd_sections, axis=1)

# Calculate the minimum and maximum Cl for each section
min_Cl_sections = np.min(Cl_sections, axis=1)
max_Cl_sections = np.max(Cl_sections, axis=1)

# Calculate the minimum and maximum Cd for each section
min_Cd_sections = np.min(Cd_sections, axis=1)
max_Cd_sections = np.max(Cd_sections, axis=1)

# Define save paths
save_path_Cl = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/average_lift_coefficient.png"
save_path_Cd = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/average_drag_coefficient.png"

# --- Plot the average Cl along the blade span ---
plt.figure(figsize=(10, 5))
plt.plot(blade_span_total_50, average_Cl_sections, 'b-o', label="Average $C_l$")
plt.fill_between(blade_span_total_50, min_Cl_sections, max_Cl_sections, color='blue', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel("Blade Span (m)", fontsize=18)
plt.ylabel("Lift Coefficient", fontsize=18)
plt.title("Lift Coefficient Along the Blade Span", fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='upper right', frameon=True)

# Save the figure with high resolution
#plt.savefig(save_path_Cl, dpi=300, bbox_inches='tight')

# Show plot
plt.show()

# --- Plot the average Cd along the blade span ---
plt.figure(figsize=(10, 5))
plt.plot(blade_span_total_50, average_Cd_sections, 'r-o', label="Average $C_d$")
plt.fill_between(blade_span_total_50, min_Cd_sections, max_Cd_sections, color='red', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel("Blade Span (m)", fontsize=18)
plt.ylabel("Drag Coefficient", fontsize=18)
plt.title("Drag Coefficient Along the Blade Span", fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='upper right', frameon=True)

# Save the figure with high resolution
#plt.savefig(save_path_Cd, dpi=300, bbox_inches='tight')

# Show plot
plt.show()
#%% lift to drag
# Calculate the lift-to-drag ratio (Cl/Cd) for each section
lift_to_drag_ratio_sections = average_Cl_sections / average_Cd_sections

# Calculate the minimum and maximum lift-to-drag ratios for each section (optional)
min_lift_to_drag_ratio_sections = min_Cl_sections / max_Cd_sections
max_lift_to_drag_ratio_sections = max_Cl_sections / min_Cd_sections

# Define the correct save path
save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/average_lift_to_drag_ratio.png"

# Create figure
plt.figure(figsize=(10, 5))

# Plot average lift-to-drag ratio
plt.plot(blade_span_total_50, lift_to_drag_ratio_sections, 'g-o', label="Average Lift-to-Drag Ratio")

# Fill between min and max range
plt.fill_between(blade_span_total_50, min_lift_to_drag_ratio_sections, max_lift_to_drag_ratio_sections, 
                 color='green', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel("Blade Span (m)", fontsize=18)
plt.ylabel("Lift-to-Drag Ratio", fontsize=18)
plt.title("Lift-to-Drag Ratio Along the Blade Span", fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='lower center', frameon=True)

# Save the figure with high resolution in the correct directory
#plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Show plot
plt.show()




#%% thrust coefficient

# Initialize arrays to store the mean, min, and max Ct for each section
mean_ct_sections = np.zeros(num_blade_sections_50)
min_ct_sections = np.zeros(num_blade_sections_50)
max_ct_sections = np.zeros(num_blade_sections_50)

# Calculate the mean, min, and max Ct for each section across the full rotation
for section in range(num_blade_sections_50):
    # Initialize a list to store Ct values for this section across the rotation
    ct_values_section = []
    
    # Loop through each blade position to get Ct values based on the wind speed
    for i in range(len(blade_positions)):
        # Interpolate Ct based on the wind speed at the current position
        ct_value = np.interp(wind_speed_hub, wind_speeds_preformance, ct_values)
        ct_values_section.append(ct_value)
    
    # Calculate the mean, min, and max Ct for this section
    mean_ct_sections[section] = np.mean(ct_values_section)
    min_ct_sections[section] = np.min(ct_values_section)
    max_ct_sections[section] = np.max(ct_values_section)

# Plot the mean Ct along the blade span with range shading
plt.figure(figsize=(10, 6))
plt.plot(blade_span_total_50, mean_ct_sections, marker='o', linestyle='-', color='purple', label='Mean Thrust Coefficient (Ct)')
plt.fill_between(blade_span_total_50, min_ct_sections, max_ct_sections, color='purple', alpha=0.2, label='Min-Max Range')

# Set labels and title
plt.xlabel('Blade Span (m)')
plt.ylabel('Thrust Coefficient (Ct)')
plt.title('Mean Thrust Coefficient (Ct) Along the Blade Span')
plt.grid(True)
plt.ylim(0, 1)
# Place the legend outside of the plot
plt.legend(loc='upper left')
#plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/22.10.figures/mean_thrust_coefficient.png', dpi=300, bbox_inches='tight')


# Show plot
plt.show()

#%% power coefficient

# Assuming `mean_axial_induction_factors` is already calculated as the average `a` for each section
# Calculate the power coefficient C_p for each section
power_coefficient_sections = 4 * mean_axial_induction_factors * (1 - mean_axial_induction_factors)**2

# Optionally calculate min and max C_p if you have min and max values for `a`
min_power_coefficient_sections = 4 * min_axial_induction_factors * (1 - min_axial_induction_factors)**2
max_power_coefficient_sections = 4 * max_axial_induction_factors * (1 - max_axial_induction_factors)**2

# Plot the power coefficient along the blade span
plt.figure(figsize=(10, 6))
plt.plot(blade_span_total_50, power_coefficient_sections, 'b-o', label='Average Power Coefficient (C_p)')
plt.fill_between(blade_span_total_50, min_power_coefficient_sections, max_power_coefficient_sections, color='blue', alpha=0.2, label='Min-Max Range')

# Set labels and title
plt.xlabel("Blade Span (m)")
plt.ylabel("Power Coefficient (C_p)")
plt.title("Power Coefficient (C_p) Along the Blade Span")
plt.grid(True)
plt.legend(loc='upper left')

# Save and show the plot
#plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/22.10.figures/power_coefficient.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the overall average power coefficient for the blade
overall_average_cp = np.mean(power_coefficient_sections)

#%%

cp_value = np.interp(wind_speed_hub, wind_speeds_preformance, cp_values)


# %% Lift and Drag Force Calculation

# Initialize arrays for lift and drag forces for each section
lift_force_sections = np.zeros_like(Cl_sections)
drag_force_sections = np.zeros_like(Cd_sections)

# Loop over each blade section to calculate lift and drag forces
for section in range(num_blade_sections_50):
    # Calculate the blade segment length for each section
    if section < num_blade_sections_50 - 1:  # Stay within bounds
        blade_segment = blade_span_total_50[section + 1] - blade_span_total_50[section]  # Segment length
    else:
        blade_segment = rotor_radius - blade_span_total_50[-1]  # Last section to rotor radius

    # Get the chord length for the corresponding section
    chord_length = blade_chord[section]

    # Reference area A = chord length * blade segment
    area = chord_length * blade_segment


    # Loop through each blade position (0° to 360°)
    for i in range(len(blade_positions)):
        # Calculate lift force: F_lift = 0.5 * rho * v_app² * Cl * A
        lift_force_sections[section, i] = 0.5 * rho * v_app_sections[section, i]**2 * Cl_sections[section, i] * area

        # Calculate drag force: F_drag = 0.5 * rho * v_app² * Cd * A
        drag_force_sections[section, i] = 0.5 * rho * v_app_sections[section, i]**2 * Cd_sections[section, i] * area

# Convert the mean lift forces to kilonewtons
mean_lift_sections = np.mean(lift_force_sections, axis=1) / 1000  # Convert from N to kN

# Keep the mean drag forces in newtons
mean_drag_sections = np.mean(drag_force_sections, axis=1)

# Calculate the minimum and maximum lift forces for each section (in kN)
min_lift_sections = np.min(lift_force_sections, axis=1) / 1000  # Convert from N to kN
max_lift_sections = np.max(lift_force_sections, axis=1) / 1000  # Convert from N to kN

# Calculate the minimum and maximum drag forces for each section (in N)
min_drag_sections = np.min(drag_force_sections, axis=1)
max_drag_sections = np.max(drag_force_sections, axis=1)

# Define correct save paths
save_path_lift = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/mean_lift_force.png"
save_path_drag = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/mean_drag_force.png"

# --- Plot Mean Lift Forces Along the Blade ---
plt.figure(figsize=(10, 5))
plt.plot(blade_span_total_50, mean_lift_sections, marker='o', linestyle='-', color='b', label='Mean Lift Force')
plt.fill_between(blade_span_total_50, min_lift_sections, max_lift_sections, color='blue', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel('Blade Span (m)', fontsize=18)
plt.ylabel('Lift Force (kN) per section', fontsize=18)
plt.title('Lift Force Along the Blade', fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='lower center', frameon=True)

# Save figure
#plt.savefig(save_path_lift, dpi=300, bbox_inches='tight')

# Show plot
plt.show()

# --- Plot Mean Drag Forces Along the Blade ---
plt.figure(figsize=(10, 5))
plt.plot(blade_span_total_50, mean_drag_sections, marker='o', linestyle='-', color='r', label='Mean Drag Force')
plt.fill_between(blade_span_total_50, min_drag_sections, max_drag_sections, color='red', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel('Blade Span (m)', fontsize=18)
plt.ylabel('Drag Force (N) per section', fontsize=18)
plt.title('Drag Force Along the Blade', fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='upper right', frameon=True)

# Save figure
#plt.savefig(save_path_drag, dpi=300, bbox_inches='tight')

# Show plot
plt.show()

#%% total lift and drag

# Initialize variables for total lift and total drag
total_lift = 0
total_drag = 0

# Loop through each blade section and calculate total lift and drag
for section in range(num_blade_sections_50):
    # Sum lift and drag contributions for all rotational positions of this section
    section_lift_sum = 0
    section_drag_sum = 0
    
    # Loop over each rotational position (100 points per rotation)
    for i in range(len(blade_positions)):
        # Lift at this rotational position
        lift = lift_force_sections[section, i]
        section_lift_sum += lift
        
        # Drag at this rotational position
        drag = drag_force_sections[section, i]
        section_drag_sum += drag
    
    # Divide the total sums by the number of positions to get the average lift and drag for this section
    section_lift_avg = section_lift_sum / len(blade_positions)
    section_drag_avg = section_drag_sum / len(blade_positions)
    
    # Add the section's average lift and drag to the total
    total_lift += section_lift_avg
    total_drag += section_drag_avg

# Convert total lift to kN (divide by 1,000)
total_lift_kN = total_lift / 1000

# Convert total drag to kN (divide by 1,000)
total_drag_kN = total_drag / 1000


# %% Normal and Tangential Force Calculation
# Initialize arrays for normal and tangential forces
P_n_sections = np.zeros_like(lift_force_sections)
P_t_sections = np.zeros_like(lift_force_sections)

# Loop through each blade section to calculate the normal and tangential forces
for section in range(num_blade_sections_50):
    # Calculate normal and tangential forces for each blade position
    for i in range(len(blade_positions)):
        # Precompute sin and cos of the inflow angle (phi)
        cos_phi = np.cos(phi_values_sections[section, i])
        sin_phi = np.sin(phi_values_sections[section, i])

        # Normal force: Lift * cos(phi) + Drag * sin(phi)
        P_n_sections[section, i] = lift_force_sections[section, i] * cos_phi + drag_force_sections[section, i] * sin_phi

        # Tangential force: Lift * sin(phi) - Drag * cos(phi)
        P_t_sections[section, i] = lift_force_sections[section, i] * sin_phi - drag_force_sections[section, i] * cos_phi

# Convert the mean, min, and max normal forces to kilonewtons (kN)
mean_P_n_sections = np.mean(P_n_sections, axis=1) / 1000
min_P_n_sections = np.min(P_n_sections, axis=1) / 1000
max_P_n_sections = np.max(P_n_sections, axis=1) / 1000

# Convert the mean, min, and max tangential forces to kilonewtons (kN)
mean_P_t_sections = np.mean(P_t_sections, axis=1) / 1000
min_P_t_sections = np.min(P_t_sections, axis=1) / 1000
max_P_t_sections = np.max(P_t_sections, axis=1) / 1000

# Define correct save paths
save_path_normal = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/mean_normal_force.png"
save_path_tangential = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/mean_tangential_force.png"

# --- Plot Mean Normal Forces Along the Blade ---
plt.figure(figsize=(10, 5))
plt.plot(blade_span_total_50, mean_P_n_sections, marker='o', linestyle='-', color='b', label='Mean Normal Force')
plt.fill_between(blade_span_total_50, min_P_n_sections, max_P_n_sections, color='blue', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel('Blade Span (m)', fontsize=18)
plt.ylabel('Normal Force (kN) per section', fontsize=18)
plt.title('Normal Force Along the Blade', fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='lower center', frameon=True)

# Save figure
#plt.savefig(save_path_normal, dpi=300, bbox_inches='tight')

# Show plot
plt.show()

# --- Plot Mean Tangential Forces Along the Blade ---
plt.figure(figsize=(10, 5))
plt.plot(blade_span_total_50, mean_P_t_sections, marker='o', linestyle='-', color='r', label='Mean Tangential Force')
plt.fill_between(blade_span_total_50, min_P_t_sections, max_P_t_sections, color='red', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel('Blade Span (m)', fontsize=18)
plt.ylabel('Tangential Force (kN) per section', fontsize=18)
plt.title('Tangential Force Along the Blade', fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='upper right', frameon=True)

# Save figure
#plt.savefig(save_path_tangential, dpi=300, bbox_inches='tight')

# Show plot
plt.show()
#%% tip loss factor
f = np.zeros((num_blade_sections_50, len(blade_positions)))  # Initialize f_tip for each section and blade position
F = np.zeros((num_blade_sections_50, len(blade_positions)))  # Initialize F_tip for each section and blade position

B = 3

# Loop through sections
for section in range(num_blade_sections_50):
    # Calculate r_central for each section (average radius of the section)
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section
    
    # Loop through blade positions for each section
    for i in range(len(blade_positions)):
        
        f[section, i] = (B / 2) * (((rotor_radius + blade_span_total_50[-1]) / 2) - r_central) / (r_central * np.sin(phi_values_sections[section, i]))  # Ensure phi is in radians if needed

        # Compute F_tip using f_tip
        F[section, i] = (2 / np.pi) * np.arccos(np.exp(-f[section, i]))
    
# Calculate the average tip loss factor F for each section
average_F_sections = np.mean(F, axis=1)

# Calculate the minimum and maximum tip loss factors for each section
min_F_sections = np.min(F, axis=1)
max_F_sections = np.max(F, axis=1)

# Define correct save path
save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/tip_loss_factor.png"

# Create figure
plt.figure(figsize=(10, 5))

# Plot average tip loss factor
plt.plot(blade_span_total_50, average_F_sections, marker='o', linestyle='-', color='b', label='Average Tip Loss Factor')
plt.fill_between(blade_span_total_50, min_F_sections, max_F_sections, color='blue', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel('Blade Span (m)', fontsize=18)
plt.ylabel('Tip Loss Factor', fontsize=18)
plt.title('Tip Loss Factor Along the Blade Span', fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='lower left', frameon=True)

# Save figure in the correct directory with high resolution
#plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Show plot
plt.show()

#%% Thrust and torque
# Initialize arrays for thrust and torque for each section
thrust_sections = np.zeros_like(P_n_sections)
torque_sections = np.zeros_like(P_t_sections)

# Calculate thrust and torque for each section
for section in range(num_blade_sections_50):
    # Calculate thrust: Thrust = P_n * B (number of blades)
    thrust_sections[section, :] = P_n_sections[section, :] * B * F[section, :]

    # Calculate r_central for each section (average radius of the section)
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section

    # Calculate torque: Torque = P_t * B * r_central
    torque_sections[section, :] = P_t_sections[section, :] * B * r_central* F[section, :]
    



# Convert the mean thrust and torque to kilonewtons and kilonewton-meters
mean_thrust_sections = np.mean(thrust_sections, axis=1) / 1000  # Convert from N to kN
mean_torque_sections = np.mean(torque_sections, axis=1) / 1000  # Convert from N·m to kN·m

# Convert the minimum and maximum thrust to kilonewtons
min_thrust_sections = np.min(thrust_sections, axis=1) / 1000  # Convert from N to kN
max_thrust_sections = np.max(thrust_sections, axis=1) / 1000  # Convert from N to kN

# Convert the minimum and maximum torque to kilonewton-meters
min_torque_sections = np.min(torque_sections, axis=1) / 1000  # Convert from N·m to kN·m
max_torque_sections = np.max(torque_sections, axis=1) / 1000  # Convert from N·m to kN·m

# Define correct save paths
save_path_thrust = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/mean_thrust.png"
save_path_torque = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/mean_torque.png"

# --- Plot Mean Thrust Along the Blade ---
plt.figure(figsize=(10, 5))
plt.plot(blade_span_total_50, mean_thrust_sections, marker='o', linestyle='-', color='b', label='Mean Thrust')
plt.fill_between(blade_span_total_50, min_thrust_sections, max_thrust_sections, color='blue', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel('Blade Span (m)', fontsize=18)
plt.ylabel('Thrust (kN) per section', fontsize=18)
plt.title('Thrust Along the Blade', fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='lower center', frameon=True)

# Save figure
#plt.savefig(save_path_thrust, dpi=300, bbox_inches='tight')

# Show plot
plt.show()

# --- Plot Mean Torque Along the Blade ---
plt.figure(figsize=(10, 5))
plt.plot(blade_span_total_50, mean_torque_sections, marker='o', linestyle='-', color='r', label='Mean Torque')
plt.fill_between(blade_span_total_50, min_torque_sections, max_torque_sections, color='red', alpha=0.2, label='Min-Max Range')

# Set labels and title with larger fonts
plt.xlabel('Blade Span (m)', fontsize=18)
plt.ylabel('Torque (kN·m) per section', fontsize=18)
plt.title('Torque Along the Blade', fontsize=20)

# Increase tick sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show grid and legend
plt.grid(True)
plt.legend(fontsize=16, loc='lower center', frameon=True)

# Save figure
#plt.savefig(save_path_torque, dpi=300, bbox_inches='tight')

# Show plot
plt.show()


#%%

# Initialize variables for total thrust and total torque
total_thrust = 0
total_torque = 0
B = 3

# Loop through each blade section and calculate total thrust and torque
for section in range(num_blade_sections_50):
    # Sum thrust contributions for all rotational positions of this section
    section_thrust_sum = 0
    section_torque_sum = 0
    
    # Loop over each rotational position (100 points per rotation)
    for i in range(len(blade_positions)):
        # Thrust at this rotational position
        thrust = P_n_sections[section, i] * B * F[section, i]
        section_thrust_sum += thrust
        
        # Calculate r_central for each section (average radius of the section)
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section
        
        # Torque at this rotational position
        torque = P_t_sections[section, i] * B * r_central * F[section, i]
        section_torque_sum += torque
    
    # Divide the total sums by 100 to get the average thrust and torque for this section
    section_thrust_avg = section_thrust_sum / len(blade_positions)
    section_torque_avg = section_torque_sum / len(blade_positions)
    
    # Add the section's average thrust and torque to the total
    total_thrust += section_thrust_avg
    total_torque += section_torque_avg

# Convert total thrust to MN (divide by 1,000,000)
total_thrust_MN = total_thrust / 1e6

# Convert total torque to MN·m (divide by 1,000,000)
total_torque_MNm = total_torque / 1e6




# %% Power Calculation



# Convert rotor speed from RPM to rad/s
omega_hub = (2 * np.pi * rot_speed_hub) / 60  # Angular velocity in rad/s

# Initialize total power variable
total_power = 0

# Loop over each blade section and calculate total power using P = Q * omega
for section in range(num_blade_sections_50):
    # Sum torque contributions for all rotational positions of this section
    section_torque_sum = 0
    
    # Loop over each rotational position (100 points per rotation)
    for i in range(len(blade_positions)):
        # Torque at this rotational position
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2 if section < num_blade_sections_50 - 1 else (rotor_radius + blade_span_total_50[-1]) / 2
        torque = P_t_sections[section, i]  * r_central * F[section, i]
        
        # Add to the section's total torque
        section_torque_sum += torque
    
    # Average torque contribution for this section over the full rotation
    section_torque_avg = section_torque_sum / len(blade_positions)
    
    # Power contribution from this section using P = Q * omega
    power_section = section_torque_avg * omega_hub
    
    # Add to the total power
    total_power += power_section

# Convert to MW for one blade
total_power_mw = total_power / 1e6

# Scale by the number of blades (B = 3)
B = 3
total_power_mw_total = total_power_mw * B


#%%

# Initialize arrays to store power contributions for each section
average_power_sections = []
min_power_sections = []
max_power_sections = []

# Loop over each blade section to calculate the average, min, and max power contributions
for section in range(num_blade_sections_50):
    # List to store power values for all rotational positions
    power_values = []
    
    # Loop over each rotational position (100 points per rotation)
    for i in range(len(blade_positions)):
        # Torque at this rotational position
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2 if section < num_blade_sections_50 - 1 else (rotor_radius + blade_span_total_50[-1]) / 2
        torque = P_t_sections[section, i] * r_central * F[section, i]
        
        # Power contribution from this rotational position
        power_value = torque * omega_hub / 1e6  # Convert to MW
        power_values.append(power_value)
    
    # Calculate average, min, and max power for this section
    average_power_sections.append(np.mean(power_values))
    min_power_sections.append(np.min(power_values))
    max_power_sections.append(np.max(power_values))

# Plot the average power generated by each section along the blade span, with min-max shading
plt.figure(figsize=(10, 6))
plt.plot(blade_span_total_50, average_power_sections, marker='o', linestyle='-', color='g', label='Average Power Contribution (MW)')
plt.fill_between(blade_span_total_50, min_power_sections, max_power_sections, color='green', alpha=0.2, label='Min-Max Range')

# Set labels and title for the plot
plt.xlabel('Blade Span (m)')
plt.ylabel('Power Contribution (MW)')
plt.title('Power Contribution by Blade Section with Min-Max Range')
plt.grid(True)
plt.legend(loc='upper left')

# Save the plot
#plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/22.10.figures/power_contribution_sections_min_max.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


#%%

def print_turbine_report(
    wind_speed_hub, rot_speed_hub, v_tip, tsr, mean_initial_wind_speed, 
    average_axial_induction_factor, mean_effective_wind_speed, average_lift_to_drag_ratio,
    overall_average_cp, total_thrust_MN, total_torque_MNm, total_power_mw, total_power_mw_total,
    total_lift_kN, total_drag_kN, ct, bladepitch, wind_speed_top, wind_speed_bottom, shear_exponent
):
    print("Wind Turbine Performance Recap", selected_time)
    print("--------------------------------")
    print(f"Wind Speed at Hub: {wind_speed_hub:.2f} m/s")
    print(f"Wind Speed at Top (H_top = {H_top} m): {wind_speed_top:.2f} m/s")
    print(f"Wind Speed at Bottom (H_bottom = {H_bottom} m): {wind_speed_bottom:.2f} m/s")
    print(f"Shear Exponent (α): {shear_exponent:.4f}")
    print(f"Average Initial Wind Speed Across Blade Span: {mean_initial_wind_speed:.2f} m/s")
    print(f"Axial Induction Factor: {average_axial_induction_factor:.4f}")
    print(f"Average Wind Speed After Axial Induction: {mean_effective_wind_speed:.2f} m/s")
    print(f"Rotor Speed Based on Hub Height Wind Speed: {rot_speed_hub:.2f} rpm")
    print(f"Tip Speed: {v_tip:.2f} m/s")
    print(f"Tip Speed Ratio (TSR): {tsr:.2f}")
    print(f"Power Coefficient (C_p): {overall_average_cp:.2f}")
    print('Ct value:', ct)
    print('Blade pitch value:', bladepitch)
    print(f"Total Lift: {total_lift_kN:.2f} kN")
    print(f"Total Drag: {total_drag_kN:.2f} kN")
    print(f"Average Lift-to-Drag Ratio Across Blade Span: {average_lift_to_drag_ratio:.2f}")
    print(f"Total Thrust: {total_thrust_MN:.4f} MN")
    print(f"Total Torque: {total_torque_MNm:.4f} MN·m")
    print(f"Total Power Output for One Blade: {total_power_mw:.2f} MW")
    print(f"Total Power Output for 3 Blades: {total_power_mw_total:.2f} MW")



print_turbine_report(
    wind_speed_hub=wind_speed_hub,
    rot_speed_hub=rot_speed_hub,
    v_tip=v_tip,
    tsr=tsr,
    mean_initial_wind_speed=np.mean(mean_initial_wind_speed),
    average_axial_induction_factor=np.mean(mean_axial_induction_factors),
    mean_effective_wind_speed=np.mean(mean_effective_wind_speed),
    average_lift_to_drag_ratio=np.mean(lift_to_drag_ratio_sections),
    overall_average_cp=overall_average_cp,
    total_thrust_MN=total_thrust_MN,
    total_torque_MNm=total_torque_MNm,
    total_power_mw=total_power_mw,
    total_power_mw_total=total_power_mw_total,
    total_lift_kN=total_lift_kN,
    total_drag_kN=total_drag_kN,
    ct=ct_values_section[1],
    bladepitch=np.mean(blade_pitch_values_sections),
    wind_speed_top=wind_speed_top,
    wind_speed_bottom=wind_speed_bottom,
    shear_exponent=shear_exponent
)
#%%
power_from_table = np.where(
    (wind_speed_hub < 3) | (wind_speed_hub > 25), 
    0, 
    np.interp(wind_speed_hub, wind_speeds_preformance, power)
)
# Beregn forskjellen
power_difference = total_power_mw_total - power_from_table

print(f"Tilsvarende effekt fra tabellen ved {wind_speed_hub} m/s er: {power_from_table:.2f} MW")
print(f"Beregnet total effekt fra modellen ved {wind_speed_hub} m/s er: {total_power_mw_total:.2f} MW")
print(f"Forskjellen mellom modellens effekt og tabellen er: {power_difference:.2f} MW")


#%% BØYMOMENT


# Filsti til ElastoDyn-bladdata
file_path_blade_data = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/IEA-15-240-RWT_ElastoDyn_blade.dat"

# Total bladmasse (kg)
total_blade_mass = 65250  
blade_length = 120  # meter

# Les inn filen og finn tabellen
with open(file_path_blade_data, "r") as file:
    lines = file.readlines()

# Finn start- og slutten av tabellen
start_index = None
for i, line in enumerate(lines):
    if "BlFract" in line and "BMassDen" in line:
        start_index = i + 2  # Tabellen starter to linjer etter dette
        break

end_index = None
for i, line in enumerate(lines[start_index:], start=start_index):
    if "BLADE MODE SHAPES" in line:  # Slutten av tabellen
        end_index = i
        break

# Les tabellen inn som en DataFrame
blade_data = pd.read_csv(
    file_path_blade_data,
    skiprows=start_index,
    nrows=end_index - start_index,
    sep=r"\s+",
    header=None,
    names=["BlFract", "PitchAxis", "StrcTwst", "BMassDen", "FlpStff", "EdgStff"],
    engine="python",
)

# Beregn lengden på hver seksjon
blade_data["Section Length"] = np.diff(blade_data["BlFract"], prepend=0) * blade_length

# Hent tetthet og lengde for hver seksjon
section_densities = blade_data["BMassDen"].values  # kg/m
section_lengths = blade_data["Section Length"].values  # meter

# Beregn massen for hver seksjon
mass_per_section = (section_lengths * section_densities) / np.sum(section_lengths * section_densities) * total_blade_mass

# Legg til massen i DataFrame
blade_data["Section Mass"] = mass_per_section





#%% FLAPWISE

# Initialize thrust (flapwise) moment array
aerodynamic_force = np.zeros((num_blade_sections_50, len(blade_positions)))

for section in range(num_blade_sections_50):
    # Mid-radius for the section
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2

    # Retrieve twist for the section (degrees → radians)
    beta_twist = np.radians(blade_twist[section])

    for i in range(len(blade_positions)):
        # Retrieve aerodynamic quantities
        F_n = P_n_sections[section, i]
        F_t = P_t_sections[section, i]
        phi = phi_values_sections[section, i]
        theta_pitch = np.radians(blade_pitch_values_sections[section, i])
        F_tip = F[section, i]

        # Effective inflow angle relative to chord
        phi_eff = phi - (theta_pitch + beta_twist)
        # Local flapwise force component
        flapwise_force = F_n * np.cos(phi_eff) - F_t * np.sin(phi_eff)

        # Flapwise bending moment = force * radius * tip loss factor
        aerodynamic_force[section, i] = flapwise_force * r_central * F_tip

# Constant for gravitational acceleration
g = 9.81  # m/s^2

# Shaft tilt angle
tilt_angle = 6  # degrees

# Cone angle
cone_angle = 4

# Initialize gravitational moment array
gravitational_moments_sections = np.zeros((len(blade_data), len(blade_positions)))
    
for section in range(len(blade_data)):
    # Mid-radius for the section (average radius)
    if section < len(blade_data) - 1:
        r_central = (blade_data["BlFract"].iloc[section] + blade_data["BlFract"].iloc[section + 1]) / 2 * blade_length
    else:
        r_central = blade_length  # Last section
            
    # Mass of the section
    m_i = blade_data["Section Mass"].iloc[section]
        
    # The gravitational moment is constant for all positions, but we iterate for structural consistency
    for i in range(len(blade_positions)):
        # Calculate gravitational moment for this position
        gravitational_moment = m_i * g * r_central * np.sin(np.radians(tilt_angle - cone_angle * np.cos(np.radians(i))))
            
        # Store the value in the array
        gravitational_moments_sections[section, i] = gravitational_moment

# Total flapwise moment = aerodynamic + gravitational contributions
total_flapwise = aerodynamic_force + gravitational_moments_sections


#%%
# Initialize array for cumulative moments (same dimensions as thrust_moment_sections)
rotor_aerodynamic = np.zeros((num_blade_sections_50, len(blade_positions)))

# Iterate backwards to compute cumulative moments
for section in range(num_blade_sections_50 - 1, -1, -1):  # From last to first section
    if section == num_blade_sections_50 - 1:  # Last section (outermost part of the blade)
        rotor_aerodynamic[section] = aerodynamic_force[section]
    else:
        rotor_aerodynamic[section] = (
            aerodynamic_force[section] + rotor_aerodynamic[section + 1]
        )

rotor_gravitational = np.zeros((len(blade_data), len(blade_positions)))

# Iterate backwards to compute cumulative gravitational moments
for section in range(len(blade_data) - 1, -1, -1):
    if section == len(blade_data) - 1:  # Last section (outermost part of the blade)
        rotor_gravitational[section] = gravitational_moments_sections[section]
    else:
        rotor_gravitational[section] = (
            gravitational_moments_sections[section] + rotor_gravitational[section + 1]
        )
        
rotor_total_flap = np.zeros((len(blade_data), len(blade_positions)))

# Iterate backwards to compute cumulative total flapwise moments
for section in range(len(blade_data) - 1, -1, -1):
    if section == len(blade_data) - 1:  # Last section (outermost part of the blade)
        rotor_total_flap[section] = total_flapwise[section]
    else:
        rotor_total_flap[section] = (
            total_flapwise[section] + rotor_total_flap[section + 1]
        )     

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(blade_positions, rotor_gravitational[0] * 1e-6, label="Gravitational", linestyle='--')
plt.plot(blade_positions, rotor_aerodynamic[0] * 1e-6, label="Aerodynamic Forces", linestyle='-.')
plt.plot(blade_positions, rotor_total_flap[0] * 1e-6, label="Total", linestyle='-')


# Plotinnstillinger
plt.xlabel("Blade Position (degrees)", fontsize=18)
plt.ylabel("(MNm)", fontsize=18)
plt.title(f"LIDAR Flapwise Bending Moment  {selected_time}", fontsize=20)
plt.legend(fontsize=17, title="Bending Moment Components", title_fontsize=18)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()


# Viser plot:
plt.show()

#%% EDGEWISE
adjusted_blade_positions = (blade_positions + 270) % 360

# Initialize array for edgewise aerodynamic bending moment (excluding gravity!)
edgewise_moments_sections = np.zeros_like(torque_sections)

# Compute edgewise bending moment for each section
for section in range(num_blade_sections_50):
    # Mid-radius for the section
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2

    # Twist for the section (degrees → radians)
    beta_twist = np.radians(blade_twist[section])

    for i in range(len(blade_positions)):
        # Pitch for the section
        phi = phi_values_sections[section, i]
        theta_pitch = np.radians(blade_pitch_values_sections[section, i])

        F_n = P_n_sections[section, i]
        F_t = P_t_sections[section, i]
        F_tip = F[section, i]

        # Effective inflow angle relative to chord
        phi_eff = phi - (theta_pitch + beta_twist)

        # Local edgewise force component (aerodynamics only)
        edgewise_force = F_t * np.cos(phi_eff) + F_n * np.sin(phi_eff)

        # Aerodynamic moment (no gravity!)
        aerodynamic_moment = edgewise_force * r_central * F_tip

        # Store aerodynamic moment in edgewise_moments_sections
        edgewise_moments_sections[section, i] = aerodynamic_moment

# Initialize array for cumulative torque moments (aerodynamic)
rotor_aerodynamic_edge = np.zeros_like(edgewise_moments_sections)

# Cumulative summation backward
for section in range(num_blade_sections_50 - 1, -1, -1):  # From last to first section
    if section == num_blade_sections_50 - 1:  # Last section
        rotor_aerodynamic_edge[section, :] = edgewise_moments_sections[section, :]
    else:
        rotor_aerodynamic_edge[section, :] = (
            edgewise_moments_sections[section, :] + rotor_aerodynamic_edge[section + 1, :]
        )

# Initialize array for gravitational moments
gravitational_moments_sections_edge = np.zeros((num_blade_sections_50, len(adjusted_blade_positions)))

# Compute gravitational moment for each section
for section in range(num_blade_sections_50):
    # Mid-radius for the section (average radius)
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section

    # Mass of the section
    m_i = blade_data["Section Mass"].iloc[section]

    # Loop through each rotor position
    for i, blade_position in enumerate(adjusted_blade_positions):
        # Calculate gravitational moment at this position
        gravitational_moment = m_i * g * r_central * np.cos(np.radians(blade_position))
        gravitational_moments_sections_edge[section, i] = gravitational_moment

# Compute cumulative gravitational moments (iterate backward)
rotor_gravity_edge = np.zeros_like(gravitational_moments_sections_edge)
for section in range(num_blade_sections_50 - 1, -1, -1):  # From last to first section
    if section == num_blade_sections_50 - 1:  # Last section
        rotor_gravity_edge[section, :] = gravitational_moments_sections_edge[section, :]
    else:
        rotor_gravity_edge[section, :] = (
            gravitational_moments_sections_edge[section, :] + rotor_gravity_edge[section + 1, :]
        )

# Combine aerodynamic and gravitational moments
cumulative_total_edgewise_moments = rotor_aerodynamic_edge + rotor_gravity_edge

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(blade_positions, rotor_gravity_edge[0] * 1e-6, label="Gravitational", linestyle='--', color='green')
plt.plot(blade_positions, rotor_aerodynamic_edge[0] * 1e-6, label="Aerodynamic Forces", linestyle='-', color='blue')
plt.plot(blade_positions, cumulative_total_edgewise_moments[0] * 1e-6, label="Total", linestyle='-.', color='red')

# Plotinnstillinger
plt.xlabel("Blade Position (degrees)", fontsize=18)
plt.ylabel("(MNm)", fontsize=18)
plt.title(f"LIDAR Edgewise Bending Moment  {selected_time}", fontsize=20)
plt.legend(fontsize=17, title="Bending Moment Components", title_fontsize=18)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()


# Vis figuren
plt.show()


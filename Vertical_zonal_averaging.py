#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 18:32:16 2025

@author: hermanellingsrud
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Vindhastigheter i området 0–25 m/s
wind_speeds = np.arange(0, 25.5, 0.5)
heights = np.array([10, 20, 50, 100, 150, 200, 250, 300])  # Høyder opp til 300m
alpha = 0.14  # Vindskjæringseksponent
hub_height = 150 
rotor_diameter = 240  # Rotor diameter (i meter)
rotor_radius = rotor_diameter / 2  # Rotor radius (halvparten av diameteren, i meter)

# Høyder for bladets topp- og bunnposisjon basert på rotorens diameter og navhøyde
H_top = hub_height + rotor_radius  # Høyde på bladspissen øverst (270 m)
H_bottom = hub_height - rotor_radius  # Høyde på bladspissen nederst (30 m)

 

# Funksjon for å beregne vindhastighet ved ulike høyder basert på navhastighet
def calculate_shear_speeds(wind_speeds, heights, hub_height, alpha):
    wind_profile = {}
    for speed in wind_speeds:
        height_speeds = speed * (heights / hub_height) ** alpha
        wind_profile[speed] = height_speeds
    return wind_profile

# Beregn vindprofilen for hver navhastighet
wind_profiles = calculate_shear_speeds(wind_speeds, heights, hub_height, alpha)

# Eksempel: print vindprofilen ved 10 m/s navhastighet
print("Vindprofil ved 10 m/s navhastighet:", wind_profiles[5])

#%%
# Velg noen spesifikke navhastigheter å plotte profiler for
selected_wind_speeds = [3, 5, 10, 15, 20, 25]  # m/s

# Opprett plott
plt.figure(figsize=(10, 6))

# Loop gjennom hver valgt vindhastighet og plot profilen
for speed in selected_wind_speeds:
    plt.plot(wind_profiles[speed], heights, label=f'{speed} m/s')

# Tilpass plottet
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Geight (m)")
plt.title(f"Windprofiles for different hub height wind speeds with shear exponent (α): {alpha}")
plt.legend(title="Hub wind speed ", loc='upper right')
plt.grid(True)

# Vis plottet
plt.show()

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
thrust = preformance_data['Thrust [MN]']
torque = preformance_data['Torque [MNm]']

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

rho = 1.225  # Lufttetthet i kg/m^3


# Vindhastighet parametre
cut_in_speed = 3.0  # Cut-in vindhastighet (i m/s)
rated_wind = 10.59  # Nominell vindhastighet (i m/s)
cut_out_speed = 25.0  # Cut-out vindhastighet (i m/s)

# Rotorhastighetsparametre
min_rot = 5.0  # Minimum rotasjonshastighet (i RPM)
rated_speed_rot = 7.56  # Nominell rotasjonshastighet (i RPM)


a_prime = 0

# Bladgeometri og posisjon
blade_positions = np.linspace(0, 360, num=360, endpoint=False)

# Høyder til bladposisjoner under rotasjon
blade_heights = hub_height + rotor_radius * np.cos(np.radians(blade_positions))

# Lister for lagring av verdier for hver vindhastighet
wind_speed_hub_list = []
rot_speed_hub_list = []
omega_hub_list = []

def calculate_wind_shear_speed(wind_speed_hub, heights, hub_height, alpha):
    # Returner en array direkte basert på input høyder
    return wind_speed_hub * (heights / hub_height) ** alpha




#%%
# Calculate wind speed at the bottom of the rotor (H_bottom) when hub wind speed is 25 m/s
hub_wind_speed = 11.0  # Hub wind speed
wind_speed_at_bottom = hub_wind_speed * (H_bottom / hub_height) ** alpha

print(f"Wind speed at the bottom of the rotor (H_bottom) when hub wind speed is {hub_wind_speed} m/s: {wind_speed_at_bottom:.2f} m/s")


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

#%% Wind for each section with wind shear using vertical zone averaging
blade_span_total = np.append(blade_span, 120)  # Extend blade span to the full rotor radius (120 meters)
num_blade_sections = len(blade_span_total)

# Heights for each section along the blade, based on span points
blade_heights_sections = np.array(blade_span_total)

# Dictionary to store wind speed rotation results for each wind speed
wind_speed_rotation_all = {}
wind_speed_hub_all = {}


# Loop over each wind speed from 0 to 25 m/s
for wind_speed in wind_speeds:
    # --- 1. Beregn vindprofil med kraftlov ---
    wind_speeds_current = wind_speed * (heights / hub_height) ** alpha

    # --- 2. Filtrer mellom 30 og 270 m ---
    mask = (heights >= 30) & (heights <= 270) & (~np.isnan(wind_speeds_current))
    heights_filtered = heights[mask]
    wind_speeds_filtered = wind_speeds_current[mask]

    # --- 3. Definer vertikale lag (3-5 soner) og beregn gjennomsnitt i hvert ---
    num_layers = 3 # change numbers of bins
    layer_edges = np.linspace(30, 270, num_layers + 1)
    layer_means = []
    for i in range(num_layers):
        in_layer = (heights_filtered >= layer_edges[i]) & (heights_filtered < layer_edges[i + 1])
        mean_speed = np.mean(wind_speeds_filtered[in_layer]) if np.any(in_layer) else 0.0
        layer_means.append(mean_speed)
    layer_means = np.array(layer_means)
    wind_speed_hub_all[wind_speed] = layer_means[1] #if 5 bins use layer_means[2], if 3 bins use layer_means[1]
    print(layer_means)



    # --- 4. Initialiser arrays ---
    blade_section_rotation = np.zeros((num_blade_sections, len(blade_positions)))
    wind_speed_rotation = np.zeros((num_blade_sections, len(blade_positions)))

    # --- 5. Beregn høyde og tilordne lagvindhastighet ---
    for section in range(num_blade_sections):
        heights_this_section = hub_height + blade_heights_sections[section] * np.cos(np.radians(blade_positions))

        # Tilordne hvert høydepunkt til en sone (lag)
        indices = np.digitize(heights_this_section, layer_edges) - 1
        indices[indices < 0] = 0
        indices[indices >= num_layers] = num_layers - 1

        wind_speeds_this_section = layer_means[indices]

        # Lagre resultatene
        blade_section_rotation[section, :] = heights_this_section
        wind_speed_rotation[section, :] = wind_speeds_this_section

    # --- 6. Lagre hele rotasjonen for denne vindhastigheten ---
    wind_speed_rotation_all[wind_speed] = wind_speed_rotation

# Velg vindhastighet og seksjon
target_wind_speed = 16.0
outer_section_index = -1  # ytterste seksjon

# Hent rotasjonen for den valgte seksjonen og vindhastigheten
wind_profile = wind_speed_rotation_all[target_wind_speed]
section_wind = wind_profile[outer_section_index, :]

# Plot vindhastighet gjennom rotasjonen
plt.figure(figsize=(8, 4))
plt.plot(blade_positions, section_wind, label=f"Wind Speed at r=120m, U_hub={target_wind_speed} m/s", color='blue')

plt.xlabel("Blade Position (°)", fontsize=12)
plt.ylabel("Wind Speed (m/s)", fontsize=12)
plt.title("Wind Speed Through Rotation — Outer Blade Section", fontsize=14)
plt.grid(True)
plt.xlim(0, 360)
plt.legend()
plt.tight_layout()
plt.show()



# Lag DataFrame
df_wind_rotation = pd.DataFrame({
    "Blade Position (°)": blade_positions,
    "Wind Speed (m/s)": section_wind
})

# Lagre til ønsket mappe
save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Baseline/Wind_rotation"
os.makedirs(save_path, exist_ok=True)

filename = f"wind_rotation_{int(num_layers)}ms.csv"
full_path = os.path.join(save_path, filename)
df_wind_rotation.to_csv(full_path, index=False)


#%%
# Verdier (kan byttes ut med dine beregnede)
layer_means = [14.42, 16.00, 16.92]  # Eksempel: U1, U2, U3
num_layers = 3
layer_edges = np.linspace(30, 270, num_layers + 1)
layer_centers = (layer_edges[:-1] + layer_edges[1:]) / 2
colors = ['#a6d9a6', '#a6b8ff', '#fce59c']  # grønn, blå, gul
  # Farger for de 3 lagene

# Lag figur
fig, ax = plt.subplots(figsize=(8.5, 8))

# Bakgrunnsfarger for hvert lag
for i in range(num_layers):
    ax.axhspan(layer_edges[i], layer_edges[i+1], color=colors[i], alpha=0.7)
    ax.text(0, layer_centers[i], f"$U_{i+1}$: {layer_means[i]:.2f} m/s",
            ha='center', va='center', fontsize=20, fontweight='bold', color='black')

# Tegn rotorskive
hub_height = 150
rotor_radius = 120
circle = plt.Circle((0, hub_height), rotor_radius, color='gray', alpha=0.15)
ax.add_patch(circle)

# Akser og formatering
ax.set_xlim(-130, 130)
ax.set_ylim(30, 270)
ax.set_xlabel("Horizontal Distance (m)", fontsize=22)
ax.set_ylabel("Height (m)", fontsize=22)
ax.tick_params(axis='both', labelsize=18)
ax.set_title(f"Bins = {num_layers}, alpha = 0.14", fontsize=25)

plt.tight_layout()
plt.show()
#%%
layer_means = [14.42, 16.00, 16.92]
num_layers = 3
layer_edges = np.linspace(30, 270, num_layers + 1)

# For step-plot: dupliser hvert lag for flat trapp
heights_step = []
wind_step = []

for i in range(num_layers):
    heights_step += [layer_edges[i], layer_edges[i+1]]
    wind_step += [layer_means[i], layer_means[i]]

# Plot step-profil
plt.figure(figsize=(5, 8))
plt.plot(wind_step, heights_step, drawstyle='steps-post', color='black', linewidth=2)

# Tilpasninger
plt.xlabel("Wind Speed (m/s)", fontsize=12)
plt.ylabel("Height (m)", fontsize=12)
plt.title("Vertical Step Wind Profile", fontsize=14)
plt.grid(True)
plt.ylim(20, 280)
plt.xlim(min(layer_means) - 0.5, max(layer_means) + 0.5)
plt.tight_layout()
plt.show()

#%% 5 bins

# layer_means = [13.7190174, 15.11705546, 16.0, 16.65756075, 17.1861583]
# num_layers = 5
# layer_edges = np.linspace(30, 270, num_layers + 1)
# layer_centers = (layer_edges[:-1] + layer_edges[1:]) / 2

# # Farger: rød, grønn, blå, gul, lilla
# colors = ['#f28e8e', '#a6d9a6', '#a6b8ff', '#fce59c', '#c8afff']

# # Figur
# fig, ax = plt.subplots(figsize=(8.5, 8))

# # Tegn fargelag og tekst
# for i in range(num_layers):
#     ax.axhspan(layer_edges[i], layer_edges[i+1], color=colors[i], alpha=0.7)
#     ax.text(0, layer_centers[i], f"$U_{{{i+1}}}$: {layer_means[i]:.2f} m/s",
#             ha='center', va='center', fontsize=20, fontweight='bold')

# # Tegn rotor-sirkel
# hub_height = 150
# rotor_radius = 120
# circle = plt.Circle((0, hub_height), rotor_radius, color='gray', alpha=0.15)
# ax.add_patch(circle)

# # Akser og formatering
# ax.set_xlim(-130, 130)
# ax.set_ylim(30, 270)
# ax.set_xlabel("Horizontal Distance (m)", fontsize=22)
# ax.set_ylabel("Height (m)", fontsize=22)
# ax.tick_params(axis='both', labelsize=18)
# ax.set_title(f"Bins = {num_layers}, alpha = 0.14", fontsize=25)
# plt.tight_layout()
# plt.savefig(f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/oppdeling_{num_layers}.pdf',dpi = 300)
# plt.show()

#%% comparing 3 and 5 bins
# # --- Profil 1: 3 lag ---
# layer_means_3 = [14.42, 16.00, 16.92]
# num_layers_3 = 3
# layer_edges_3 = np.linspace(30, 270, num_layers_3 + 1)

# heights_step_3 = []
# wind_step_3 = []
# for i in range(num_layers_3):
#     heights_step_3 += [layer_edges_3[i], layer_edges_3[i+1]]
#     wind_step_3 += [layer_means_3[i], layer_means_3[i]]

# # --- Profil 2: 5 lag ---
# layer_means_5 = [13.72, 15.12, 16.00, 16.66, 17.19]
# num_layers_5 = 5
# layer_edges_5 = np.linspace(30, 270, num_layers_5 + 1)

# heights_step_5 = []
# wind_step_5 = []
# for i in range(num_layers_5):
#     heights_step_5 += [layer_edges_5[i], layer_edges_5[i+1]]
#     wind_step_5 += [layer_means_5[i], layer_means_5[i]]

# # --- Plot begge profiler ---
# plt.figure(figsize=(5, 8))
# plt.plot(wind_step_3, heights_step_3, drawstyle='steps-post', label='3 Bins', color='black', linewidth=2)
# plt.plot(wind_step_5, heights_step_5, drawstyle='steps-post', label='5 Bins', color='blue', linewidth=2, linestyle='--')

# # Tilpasninger
# plt.xlabel("Wind Speed (m/s)", fontsize=16)
# plt.ylabel("Height (m)", fontsize=16)
# plt.title("Vertical Step Wind Profiles", fontsize=20)
# plt.grid(True)
# plt.legend(fontsize=15)
# plt.ylim(20, 280)
# plt.xlim(min(wind_step_5 + wind_step_3) - 0.5, max(wind_step_5 + wind_step_3) + 0.5)
# plt.tight_layout()
# plt.savefig(f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/step_profile_3_5.pdf',dpi = 300)

# plt.show()


#%%
# Loop over hver vindhastighet fra 0 til 25 m/s
for wind_speed in wind_speed_hub_all:
    # Sett navhastighet og lagre for denne vinden
    wind_speed_hub = wind_speed  # A3 = midtsonen

    wind_speed_hub_list.append(wind_speed_hub)

    # Beregn rotorhastighet, satt til 0 utenfor cut-in/cut-out
    rot_speed_hub_array = np.where(
        (wind_speed_hub < cut_in_speed) | (wind_speed_hub > cut_out_speed), 
        0, 
        np.interp(wind_speed_hub, wind_speeds_preformance, rotor_speeds)
    )
    
    rot_speed_hub = float(rot_speed_hub_array) if rot_speed_hub_array.size == 1 else rot_speed_hub_array
    rot_speed_hub_list.append(rot_speed_hub)

    # Konverter rotorhastighet fra RPM til rad/s og lagre
    omega_hub = (2 * np.pi * rot_speed_hub) / 60
    omega_hub_list.append(omega_hub)

    # Beregn vindskjærhastighet langs hele bladet for denne navhastigheten
    wind_shear_speeds = calculate_wind_shear_speed(wind_speed_hub, blade_heights, hub_height, alpha)

    # Eksempelvis kan du nå bruke `wind_shear_speeds` i videre beregninger
    print(f"Vindhastigheter langs bladet ved {wind_speed} m/s navhastighet:", wind_shear_speeds)


#%%

blade_span_total_50 = blade_span_total[:50]  # Keep only the first 50 sections
num_blade_sections_50 = len(blade_span_total_50)

# Create dictionaries to store results for each wind speed
phi_values_all = {}
w_sections_all = {}
rot_speed_sections_all = {}
omega_sections_all = {}
u_sections_all = {}
v_app_sections_all = {}
a_sections_all = {}



for wind_speed in wind_speed_hub_all:
    # Hent rotasjonshastigheter og beregn omega
    wind_speed_hub = wind_speed  # A3 = midtsonen

    rot_speed_hub = np.where(
        (wind_speed_hub < cut_in_speed) | (wind_speed_hub > cut_out_speed), 
        0, 
        np.interp(wind_speed_hub, wind_speeds_preformance, rotor_speeds)
    )
    omega_hub = (2 * np.pi * rot_speed_hub) / 60  # Konverter RPM til rad/s

    # Initialiser arrays for denne vindhastigheten
    phi_values_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    w_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    rot_speed_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    omega_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    u_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    v_app_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    a_sections = np.zeros((num_blade_sections_50, len(blade_positions)))

    # Få vindhastighet gjennom rotasjonen med vindskjærprofil
    wind_speed_rotation = wind_speed_rotation_all[wind_speed]  # Vindhastigheter langs bladet for denne vindhastigheten

    for section in range(num_blade_sections_50):
        # Kalkuler r_central for hver seksjon
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Siste seksjon

        for i in range(len(blade_positions)):
            # Bruk vindskjær-vindhastigheten i stedet for homogen vindhastighet
            wind_speed_current = wind_speed_rotation[section, i]

            # Beregn aksial induksjonsfaktor `a`
            a_section = get_axial_induction_factor(wind_speed_hub, wind_speeds_preformance, ct_values)
            a_sections[section, i] = a_section

            # Beregn effektiv vindhastighet med aksial induksjon
            u_sections[section, i] = wind_speed_current * (1 - a_section)
            
            # Lagre hub rotasjonshastighet og omega
            rot_speed_sections[section, i] = rot_speed_hub
            omega_sections[section, i] = omega_hub

            # Beregn tangentialhastighet med tangential induksjonsfaktor
            w_sections[section, i] = omega_sections[section, i] * r_central * (1 + a_prime)

            # Beregn innstrømningsvinkel (phi)
            phi_values_sections[section, i] = np.arctan(u_sections[section, i] / w_sections[section, i])

            # Beregn tilsynelatende vindhastighet
            v_app_sections[section, i] = np.sqrt(u_sections[section, i]**2 + w_sections[section, i]**2)

    # Lagre resultatene for denne vindhastigheten
    phi_values_all[wind_speed] = phi_values_sections
    w_sections_all[wind_speed] = w_sections
    rot_speed_sections_all[wind_speed] = rot_speed_sections
    omega_sections_all[wind_speed] = omega_sections
    u_sections_all[wind_speed] = u_sections
    v_app_sections_all[wind_speed] = v_app_sections
    a_sections_all[wind_speed] = a_sections




#%% ordinær
# Create a dictionary to store blade pitch values for each wind speed
blade_pitch_values_all = {}

# Loop over each wind speed in the range (0 to 25 m/s)
for wind_speed in wind_speed_hub_all:
    # Set the homogeneous wind speed for this iteration
    wind_speed_hub = wind_speed

    # Interpolate the blade pitch for this wind speed based on the performance data
    blade_pitch_value = np.interp(wind_speed_hub, wind_speeds_preformance, pitch_angles)

    # Initialize a 2D array for blade pitch values at each section during the full rotation
    blade_pitch_values_sections = np.zeros((num_blade_sections_50, len(blade_positions)))

    # Loop through each blade section and set the interpolated pitch value
    for section in range(num_blade_sections_50):
        # Set the pitch value for each rotation angle at this section
        blade_pitch_values_sections[section, :] = blade_pitch_value

    # Store the blade pitch values for this wind speed in the dictionary
    blade_pitch_values_all[wind_speed] = blade_pitch_values_sections

# blade_pitch_values_all[12] = blade_pitch_values_all[12] + 0.02
# blade_pitch_values_all[12.5] = blade_pitch_values_all[12.5] - 0.04
# blade_pitch_values_all[13] = blade_pitch_values_all[13] + 0.02
# blade_pitch_values_all[20] = blade_pitch_values_all[20] + 0.005

#%%
alpha_values_all = {}

# Loop over each wind speed in the range (0 to 25 m/s)
for wind_speed in wind_speeds:
    # Retrieve or calculate values for the current wind speed
    phi_values_sections = phi_values_all[wind_speed]
    blade_pitch_values_sections = blade_pitch_values_all[wind_speed]
    
    # Initialize a 2D array for angle of attack (alpha) for each section during the rotation
    alpha_sections = np.zeros_like(phi_values_sections)

    # Loop through each section to calculate the angle of attack (alpha)
    for section in range(num_blade_sections_50):
        # Convert inflow angle (phi) from radians to degrees for the current section
        phi_deg_sections = np.degrees(phi_values_sections[section, :])
        
        # Calculate the angle of attack: alpha = phi (deg) - blade pitch (deg) - blade twist (deg)
        alpha_sections[section, :] = phi_deg_sections - blade_pitch_values_sections[section, :] - blade_twist[section]

    # Store the alpha values for this wind speed in the dictionary
    alpha_values_all[wind_speed] = alpha_sections
    
print(alpha_values_all[25])


#%%
# Dictionary to store Cl and Cd values for each wind speed
Cl_values_all = {}
Cd_values_all = {}

# Loop over each wind speed in the range (0 to 25 m/s)
for wind_speed in wind_speeds:
    # Initialize arrays for Cl and Cd for each section for this wind speed
    Cl_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    Cd_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    
    # Get the alpha values for each section and position for this wind speed
    alpha_sections = alpha_values_all[wind_speed]
    
    # Loop over each blade section
    for section in range(num_blade_sections_50):
        # Retrieve the airfoil data for Cl, Cd, and aoa for this section
        Cl_current = cl[section]  # Lift coefficient values for this section's airfoil
        Cd_current = cd[section]  # Drag coefficient values for this section's airfoil
        aoa_current = aoa[section]  # Angle of attack values for this section's airfoil

        # Loop through each position for the current section
        for i in range(len(blade_positions)):
            # Clip the angle of attack to fit within the range of aoa_current
            alpha_clipped = np.clip(alpha_sections[section, i], aoa_current.min(), aoa_current.max())
            
            # Interpolate Cl and Cd based on the clipped alpha for this section and position
            Cl_sections[section, i] = np.interp(alpha_clipped, aoa_current, Cl_current)
            Cd_sections[section, i] = np.interp(alpha_clipped, aoa_current, Cd_current)

    # Store the detailed Cl and Cd arrays for each section and position for this wind speed
    Cl_values_all[wind_speed] = Cl_sections
    Cd_values_all[wind_speed] = Cd_sections
    


#%%
# Dictionaries to store lift and drag forces for each wind speed
lift_force_all = {}
drag_force_all = {}

# Loop over each wind speed from 0 to 25 m/s
for wind_speed in wind_speeds:
    # Get the Cl, Cd, and v_app values for the current wind speed
    Cl_sections = Cl_values_all[wind_speed]
    Cd_sections = Cd_values_all[wind_speed]
    v_app_sections = v_app_sections_all[wind_speed]
    
    # Initialize arrays for lift and drag forces for each section and position
    lift_force_sections = np.zeros_like(Cl_sections)
    drag_force_sections = np.zeros_like(Cd_sections)

    # Loop over each blade section
    for section in range(num_blade_sections_50):
        # Calculate the blade segment length for each section
        if section < num_blade_sections_50 - 1:
            blade_segment = blade_span_total_50[section + 1] - blade_span_total_50[section]
        else:
            blade_segment = rotor_radius - blade_span_total_50[-1]

        # Get the chord length for the current section
        chord_length = blade_chord[section]

        # Reference area A = chord length * blade segment
        area = chord_length * blade_segment

        # Loop through each blade position (0° to 360°)
        for i in range(len(blade_positions)):
            # Calculate lift force: F_lift = 0.5 * rho * v_app² * Cl * A
            lift_force_sections[section, i] = 0.5 * rho * v_app_sections[section, i]**2 * Cl_sections[section, i] * area

            # Calculate drag force: F_drag = 0.5 * rho * v_app² * Cd * A
            drag_force_sections[section, i] = 0.5 * rho * v_app_sections[section, i]**2 * Cd_sections[section, i] * area

    # Store the lift and drag forces for this wind speed in the dictionaries
    lift_force_all[wind_speed] = lift_force_sections
    drag_force_all[wind_speed] = drag_force_sections
    

#%%


# Dictionaries to store normal and tangential forces for each wind speed
P_n_all = {}
P_t_all = {}

# Loop over each wind speed in the range (0 to 25 m/s)
for wind_speed in wind_speeds:
    # Get lift and drag forces and phi values for the current wind speed
    lift_force_sections = lift_force_all[wind_speed]
    drag_force_sections = drag_force_all[wind_speed]
    phi_values_sections = phi_values_all[wind_speed]
    
    # Initialize arrays for normal and tangential forces for each section and position
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
            P_n_sections[section, i] = (
                lift_force_sections[section, i] * cos_phi + drag_force_sections[section, i] * sin_phi
            )

            # Tangential force: Lift * sin(phi) - Drag * cos(phi)
            P_t_sections[section, i] = (
                lift_force_sections[section, i] * sin_phi - drag_force_sections[section, i] * cos_phi
            )

    # Store the normal and tangential forces for this wind speed in the dictionaries
    P_n_all[wind_speed] = P_n_sections
    P_t_all[wind_speed] = P_t_sections

#%%

# Dictionary to store tip loss factor F for each wind speed
tip_loss_factor_all = {}

# Set number of blades
B = 3  # Adjust if the turbine model requires more blades

# Loop over each wind speed in the range (0 to 25 m/s)
for wind_speed in wind_speeds:
    # Retrieve the inflow angle (phi) values for the current wind speed
    phi_values_sections = phi_values_all[wind_speed]
    
    # Initialize arrays for f and F for each section and position
    f = np.zeros((num_blade_sections_50, len(blade_positions)))
    F = np.zeros((num_blade_sections_50, len(blade_positions)))

    # Loop through each section
    for section in range(num_blade_sections_50):
        # Calculate r_central for each section (average radius of the section)
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section
        
        # Loop through blade positions for each section
        for i in range(len(blade_positions)):
            # Calculate f_tip based on the inflow angle phi and central radius
            f[section, i] = (B / 2) * (((rotor_radius + blade_span_total_50[-1]) / 2) - r_central) / (r_central * np.sin(phi_values_sections[section, i]))

            # Compute F_tip using f_tip
            F[section, i] = (2 / np.pi) * np.arccos(np.exp(-f[section, i]))

    # Store the tip loss factor F for this wind speed in the dictionary
    tip_loss_factor_all[wind_speed] = F
    
#%%

# Dictionaries to store thrust and torque for each wind speed
thrust_all = {}
torque_all = {}

# Number of blades
B = 1  # Adjust if needed

# Loop over each wind speed in the range (0 to 25 m/s)
for wind_speed in wind_speeds:
    # Retrieve normal and tangential forces, and tip loss factor for the current wind speed
    P_n_sections = P_n_all[wind_speed]
    P_t_sections = P_t_all[wind_speed]
    F_sections = tip_loss_factor_all[wind_speed]
    
    # Initialize arrays for thrust and torque for each section and position
    thrust_sections = np.zeros_like(P_n_sections)
    torque_sections = np.zeros_like(P_t_sections)

    # Calculate thrust and torque for each section
    for section in range(num_blade_sections_50):
        # Calculate thrust: Thrust = P_n * B * F
        thrust_sections[section, :] = P_n_sections[section, :] * B * F_sections[section, :]

        # Calculate r_central for each section (average radius of the section)
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section

        # Calculate torque: Torque = P_t * B * r_central * F
        torque_sections[section, :] = P_t_sections[section, :] * B * r_central * F_sections[section, :]

    # Store the thrust and torque for this wind speed in the dictionaries
    thrust_all[wind_speed] = thrust_sections
    torque_all[wind_speed] = torque_sections
    
# Initialize dictionaries to store total thrust and total torque for each wind speed
total_thrust_all = {}
total_torque_all = {}

# Number of blades
B = 3  # Set to 3 blades for the total calculations

# Loop over each wind speed in the range (0 to 25 m/s)
for wind_speed in wind_speeds:
    # Retrieve normal and tangential forces, and tip loss factor for the current wind speed
    P_n_sections = P_n_all[wind_speed]
    P_t_sections = P_t_all[wind_speed]
    F_sections = tip_loss_factor_all[wind_speed]
    
    # Initialize variables for total thrust and total torque for the current wind speed
    total_thrust = 0
    total_torque = 0

    # Loop through each blade section to calculate thrust and torque
    for section in range(num_blade_sections_50):
        # Sum thrust and torque contributions over all rotational positions for this section
        section_thrust_sum = 0
        section_torque_sum = 0
        
        # Loop over each rotational position (100 points per rotation)
        for i in range(len(blade_positions)):
            # Thrust at this rotational position
            thrust = P_n_sections[section, i] * B * F_sections[section, i]
            section_thrust_sum += thrust
            
            # Calculate r_central for each section (average radius of the section)
            if section < num_blade_sections_50 - 1:
                r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
            else:
                r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section
            
            # Torque at this rotational position
            torque = P_t_sections[section, i] * B * r_central * F_sections[section, i]
            section_torque_sum += torque
        
        # Calculate the average thrust and torque for this section
        section_thrust_avg = section_thrust_sum / len(blade_positions)
        section_torque_avg = section_torque_sum / len(blade_positions)
        
        # Add the section's average thrust and torque to the total
        total_thrust += section_thrust_avg
        total_torque += section_torque_avg

    # Convert total thrust and torque to MN and MN·m
    total_thrust_MN = total_thrust / 1e6
    total_torque_MNm = total_torque / 1e6
    
    # Store the total thrust and torque for this wind speed in the dictionaries
    total_thrust_all[wind_speed] = total_thrust_MN
    total_torque_all[wind_speed] = total_torque_MNm

#%%

   # Dictionary to store total power output (in MW) for each wind speed
power_output_all = {}

# Set the number of blades
B = 3  # Number of blades

# Loop over each wind speed
for wind_speed in wind_speeds:
    if wind_speed == 0.0:
        # Set power output to 0 for 0 m/s wind speed directly
        power_output_all[wind_speed] = 0
        continue
    elif wind_speed <= 2.5 or wind_speed > 25:
        # Set rotor speed and omega to 0 for wind speeds <= 2.5 m/s and over cut-out
        rot_speed_hub = 0
        omega_hub = 0
    else:
        # Convert rotor speed from RPM to rad/s for this wind speed
        rot_speed_hub = np.interp(wind_speed, wind_speeds_preformance, rotor_speeds)
        omega_hub = (2 * np.pi * rot_speed_hub) / 60  # Angular velocity in rad/s

    # Initialize total power variable for this wind speed
    total_power = 0

    # Loop over each blade
    for blade_offset in [0, 120, 240]:  # Blade positions (0°, 120°, 240°)
        # Loop over each blade section
        for section in range(num_blade_sections_50):
            # Sum torque contributions for all rotational positions of this section
            section_torque_sum = 0

            # Loop over each rotational position
            for i, blade_position in enumerate(blade_positions):
                # Adjust blade position for the current blade
                adjusted_position = (blade_position + blade_offset) % 360

                # Find the nearest rotor index for this adjusted position
                rotor_index = (np.abs(blade_positions - adjusted_position)).argmin()

                # Torque at this rotational position
                r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2 if section < num_blade_sections_50 - 1 else (rotor_radius + blade_span_total_50[-1]) / 2
                torque = P_t_all[wind_speed][section, rotor_index] * r_central * tip_loss_factor_all[wind_speed][section, rotor_index]

                # Add to the section's total torque
                section_torque_sum += torque

            # Average torque contribution for this section over the full rotation
            section_torque_avg = section_torque_sum / len(blade_positions)

            # Power contribution from this section using P = Q * omega
            power_section = section_torque_avg * omega_hub

            # Add to the total power for this blade
            total_power += power_section

    # Convert to MW for all blades
    total_power_mw_total = total_power / 1e6

    # Store the total power output for this wind speed
    power_output_all[wind_speed] = total_power_mw_total


# Extract the power curve from the rotor performance data
wind_speeds_performance = preformance_data['Wind [m/s]'].values
power_performance = preformance_data['Power [MW]'].values

# Plot the calculated power output curve and the rotor performance power curve
plt.figure(figsize=(10, 6))

# Plot calculated power curve
plt.plot(list(power_output_all.keys()), list(power_output_all.values()), marker='o', linestyle='-', color='blue', label="Calculated Power Output")

# Plot rotor performance power curve
plt.plot(wind_speeds_performance, power_performance, marker='x', linestyle='--', color='red', label="Rotor Performance Power Output")

# Add labels and title
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power Output (MW)")
plt.title(f"Power Output Curve with Wind Shear (α = {alpha})")
plt.legend()
plt.grid(True)
plt.ylim(-1, 16)

# Show the plot
plt.show()

for wind_speed in wind_speeds:
    # Calculate power from the table for the current wind speed
    power_from_table = np.where(
        (wind_speed < 3) | (wind_speed > 25), 
        0, 
        np.interp(wind_speed, wind_speeds_performance, power_performance)
    )
    
    # Calculate the difference for the current wind speed
    power_difference = power_output_all[wind_speed] - power_from_table

    # Print the results for the current wind speed
    print(
        f"Wind Speed: {wind_speed} m/s, "
        f"Total Power: {power_output_all[wind_speed]:.4f} MW, "
        f"Power difference from expected outcome: {power_difference:.4f} MW"
    )


#%%

# Konverter dictionary til DataFrame
power_output_df = pd.DataFrame({
    'Wind Speed (m/s)': list(power_output_all.keys()),
    'Calculated Power Output (MW)': list(power_output_all.values())
})

# Angi lagringssti
save_path_power = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Baseline/Power/calculated_power_output_{num_layers}.csv'

# Lagre til CSV
power_output_df.to_csv(save_path_power, index=False)
print(f"Saved power output to: {save_path_power}")

#%%
# Dictionary for å lagre total effekt per vindhastighet
power_output_all = {}

# Beregninger
for wind_speed in wind_speeds:
    if wind_speed == 0.0:
        # Sett effekt til 0 for 0 m/s vindhastighet
        power_output_all[wind_speed] = [0] * len(blade_positions)
        continue
    elif wind_speed <= 2.5 or wind_speed > 25:
        # Sett rotorhastighet og omega til 0 for vindhastigheter <= 2.5 m/s eller over cut-out
        rot_speed_hub = 0
        omega_hub = 0
    else:
        # Interpoler rotorhastighet og regn ut omega (vinkelhastighet)
        rot_speed_hub = np.interp(wind_speed, wind_speeds_preformance, rotor_speeds)
        omega_hub = (2 * np.pi * rot_speed_hub) / 60  # Vinkelhastighet i rad/s

    # Beregn total effekt
    total_power_per_position = []
    for i in range(len(blade_positions)):
        total_power = 0
        for section in range(num_blade_sections_50):
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2 if section < num_blade_sections_50 - 1 else (rotor_radius + blade_span_total_50[-1]) / 2
            torque = P_t_all[wind_speed][section, i] * r_central * tip_loss_factor_all[wind_speed][section, i]
            power_section = torque * omega_hub
            total_power += power_section
        total_power_per_position.append(total_power / 1e6 )  # Konverter til MW og skaler med antall blader
    power_output_all[wind_speed] = total_power_per_position

# Plotting
plt.figure(figsize=(10, 5))
for wind_speed in [11, 15, 20, 25]:  # Select some wind speeds for plotting
    plt.plot(
        blade_positions,  # Convert to degrees
        power_output_all[wind_speed],
        label=f"{wind_speed} m/s"
    )

# Adding vertical dashed lines at x = 90 and x = 270
plt.axvline(x=90, color='black', linestyle='--', linewidth=1)
plt.axvline(x=270, color='black', linestyle='--', linewidth=1)

# Adding axis labels, title, and legend
plt.xlabel("(degrees)", fontsize=18)
plt.ylabel("(MW)", fontsize=18)
plt.title(f"Power Output Across Blade Positions (Bins = {num_layers})", fontsize=20)

# Adding a legend with a title
legend = plt.legend(
    title="Hub Wind Speed", 
    fontsize=16, 
    title_fontsize=16, 
    loc='center left', 
    bbox_to_anchor=(1, 0.5)
)

plt.setp(legend.get_title(), fontsize=16)  # Adjust legend title font size
# Adjust tick parameters for larger ticks
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=14)

# Adding grid and showing the plot
plt.grid(True)

save_path = f"/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/power_output_blade_positions_{num_layers}.pdf"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

#%%
# For valgt vindhastighet (f.eks. 11 m/s)
target_wind_speed = 11.0

# Hent omega
rot_speed_hub = np.interp(target_wind_speed, wind_speeds_preformance, rotor_speeds)
omega_hub = (2 * np.pi * rot_speed_hub) / 60

# Initialiser array for effekt per rotasjonsposisjon
power_rotation = np.zeros(len(blade_positions))

# Loop over rotasjonsposisjoner (0–359°)
for i, blade_position in enumerate(blade_positions):
    total_power_this_pos = 0

    # Loop over alle tre blader
    for blade_offset in [0, 120, 240]:
        adjusted_position = (blade_position + blade_offset) % 360
        rotor_index = (np.abs(blade_positions - adjusted_position)).argmin()

        # Loop over seksjoner
        for section in range(num_blade_sections_50):
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2 if section < num_blade_sections_50 - 1 else (rotor_radius + blade_span_total_50[-1]) / 2
            torque = P_t_all[target_wind_speed][section, rotor_index] * r_central * tip_loss_factor_all[target_wind_speed][section, rotor_index]
            power = torque * omega_hub
            total_power_this_pos += power

    power_rotation[i] = total_power_this_pos / 1e6  # konverter til MW

# Plot
plt.figure(figsize=(8, 4))
plt.plot(blade_positions, power_rotation, label=f"{target_wind_speed} m/s", color='tab:blue')
plt.xlabel("Rotor Position (°)", fontsize=12)
plt.ylabel("Power Output (MW)", fontsize=12)
plt.title("Total Power from All Blades Through a Full Rotation", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import os

# Lag DataFrame med posisjon og effekt
df_power_rotation = pd.DataFrame({
    "Blade Position (°)": blade_positions,
    "Power Output (MW)": power_rotation
})

# Definer lagringsbane og lagre
save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Baseline/Power_rotation"
os.makedirs(save_path, exist_ok=True)  # sørg for at mappen finnes

filename = f"power_rotation_{int(num_layers)}ms.csv"
df_power_rotation.to_csv(os.path.join(save_path, filename), index=False)



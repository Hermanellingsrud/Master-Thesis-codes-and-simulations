#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 09:40:53 2025

@author: hermanellingsrud
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

preformance_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Alle_verdier/IEA-15-240-RWT_tabular/Rotor Performance-Table 1.csv'

preformance_data = pd.read_csv(preformance_path, sep=';', decimal=',')

wind_speeds_preformance = preformance_data['Wind [m/s]']
pitch_angles = preformance_data['Pitch [deg]']
rotor_speeds = preformance_data['Rotor Speed [rpm]']
ct_values = preformance_data['Thrust Coefficient [-]']
cp_values = preformance_data['Aero Power Coefficient [-]']
power = preformance_data['Power [MW]']

# Rotasjonsvinkler (0–360 grader)
blade_positions = np.linspace(0, 360, num=360, endpoint=False)

# 1. Definert hub-vind
ref_wind_speed = 15
wind_speed_hub = ref_wind_speed

# 2. Interpoler rotorhastighet (RPM) fra ytelsesdata
rot_speed_hub = np.where(
    (wind_speed_hub < 3) | (wind_speed_hub > 25), 
    0, 
    np.interp(wind_speed_hub, wind_speeds_preformance, rotor_speeds)
)

# 3. Beregn omega (rad/s)
omega_hub = (2 * np.pi * rot_speed_hub) / 60

# 4. Perioden for én rotasjon i sekunder:
T_rot = 2 * np.pi / omega_hub *15 # eller: 60 / RPM

# 5. Tidspunktene gjennom én rotasjon
dt = 0.1  # tidssteg i sekunder (kan settes til f.eks. 0.1 for høyere oppløsning)
time_vector = np.arange(0, T_rot, dt)

# 6. Rotasjonsvinkel som funksjon av tid
theta_deg = (omega_hub * time_vector * 180 / np.pi) % 360

print(time_vector)
print(theta_deg)

# Parametere
alpha = 0.14
ref_height = 150

TI = 0.05

# Høyder (bunn til topp av rotor)
heights = np.linspace(ref_height - 120, ref_height + 120, 30)

# Parametere
hub_height = 150
rotor_radius = 120


#%%

# --- Generer én temporalt og vertikalt korrelert turbulent profil for hvert tidssteg ---
turbulent_profiles = []

# Midlere profil og standardavvik
u_bar = ref_wind_speed * (heights / hub_height) ** alpha
phi = 0.95  # Tidskorrelasjon (0 = uavhengig, 1 = helt avhengig)
sigma_noise = TI * u_bar * np.sqrt(1 - phi**2)


# Startprofil (nullavvik)
prev_turb = np.zeros_like(u_bar)

#np.random.seed(4)  # Valgfritt – for reproduserbarhet
# --- Lag én temporalt og vertikalt korrelert turbulent profil per tidssteg basert på IEC-koherens ---
def compute_iec_coherence_matrix(z_levels, U, f, L_k):
    z_levels = np.array(z_levels)
    dz_matrix = np.abs(z_levels[:, None] - z_levels[None, :])
    term1 = (f * dz_matrix / U)**2
    term2 = (0.12 * dz_matrix / L_k)**2
    return np.exp(-12 * np.sqrt(term1 + term2))

# IEC-koherensparametre
U = ref_wind_speed
f = 0.1  # antatt dominant frekvens
L_k = 8.1 * 42  # basert på IEC (Lambda1 ≈ 42 m)

coherence_matrix = compute_iec_coherence_matrix(heights, U, f, L_k)

turbulent_profiles = []
prev_turb = np.zeros_like(u_bar)

for _ in range(len(time_vector)):
    noise = np.random.normal(loc=0, scale=sigma_noise)
    turb = phi * prev_turb + noise
    turb_corr = coherence_matrix @ turb / np.sum(coherence_matrix, axis=1)
    profile = np.clip(u_bar + turb_corr, 0.1, None)
    turbulent_profiles.append(profile)
    prev_turb = turb


plt.figure(figsize=(6, 8))

# Plot alle profiler med lav gjennomsiktighet
for profile in turbulent_profiles:
    plt.plot(profile, heights, alpha=0.2, color='steelblue')

# Beregn og plott gjennomsnittlig profil
mean_profile = np.mean(turbulent_profiles, axis=0)
plt.plot(mean_profile, heights, color='darkred', linewidth=2.5, label='Average profile')

# Plotinnstillinger
plt.xlabel("Wind Speed [m/s]", fontsize=14)
plt.ylabel("Height [m]", fontsize=14)
plt.title("Turbulent Vertical Wind Profiles over One Rotation", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=13)

plt.tight_layout()
plt.show()


# --- Vind langs bladet under rotasjon ---
U_rotation = np.zeros_like(time_vector)

for i, theta in enumerate(theta_deg):
    blade_height = hub_height + rotor_radius * np.cos(np.radians(theta))
    U_rotation[i] = np.interp(blade_height, heights, turbulent_profiles[i])

# --- Plot vind langs bladets rotasjon ---
plt.figure(figsize=(10, 4))
plt.plot(time_vector, U_rotation, label='Vind langs bladets bane (turbulent)')
plt.xlabel('time [s]')
plt.ylabel('Vindhastighet [m/s]')
plt.title('Tidsvarierende vindhastighet opplevd av bladet under en rotasjon')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%%
sigma_uncorrelated = TI * u_bar
# --- 1. Ukorrelert turbulens ---
turbulent_profiles_uncorrelated = []
for i in range(len(time_vector)):
#    np.random.seed(i + 4)
    turb = np.random.normal(loc=0, scale=sigma_uncorrelated)
    profile = np.clip(u_bar + turb, 0.1, None)
    turbulent_profiles_uncorrelated.append(profile)

# --- 2. Tidskorrelert turbulens ---
turbulent_profiles_timecorr = []
phi = 0.95
sigma_noise = TI * u_bar * np.sqrt(1 - phi**2)
prev_turb = np.zeros_like(u_bar)

for _ in range(len(time_vector)):
    noise = np.random.normal(loc=0, scale=sigma_noise)
    turb = phi * prev_turb + noise
    profile = np.clip(u_bar + turb, 0.1, None)
    turbulent_profiles_timecorr.append(profile)
    prev_turb = turb

# --- 3. Tids- og høydekorrelert turbulens basert på IEC-koherens ---
def compute_iec_coherence_matrix(z_levels, U, f, L_k):
    """
    Beregn vertikal koherensmatrise basert på IEC-formelen.
    """
    z_levels = np.array(z_levels)
    dz_matrix = np.abs(z_levels[:, None] - z_levels[None, :])
    term1 = (f * dz_matrix / U) ** 2
    term2 = (0.12 * dz_matrix / L_k) ** 2
    coherence_matrix = np.exp(-12 * np.sqrt(term1 + term2))
    return coherence_matrix

# Parametre for IEC-koherens
U = ref_wind_speed         # Midlere vind
f = 0.1                    # Antatt dominant frekvens (Hz)
L_k = 8.1 * 42             # IEC: L_k = 8.1 * Λ1 (Λ1 ≈ 42 m for offshore)

coherence_matrix = compute_iec_coherence_matrix(heights, U, f, L_k)

# Generer profiler
turbulent_profiles_bothcorr = []
prev_turb = np.zeros_like(u_bar)

for _ in range(len(time_vector)):
    noise = np.random.normal(loc=0, scale=sigma_noise)
    turb = phi * prev_turb + noise

    # IEC-koherensbasert vertikal korrelasjon
    turb_corr = coherence_matrix @ turb / np.sum(coherence_matrix, axis=1)

    profile = np.clip(u_bar + turb_corr, 0.1, None)
    turbulent_profiles_bothcorr.append(profile)
    prev_turb = turb


# --- Plot alle tre ---
fig, axs = plt.subplots(1, 3, figsize=(18, 8), sharey=True)

# Uncorrelated
for prof in turbulent_profiles_uncorrelated:
    axs[0].plot(prof, heights, alpha=0.2, color='steelblue')
axs[0].plot(np.mean(turbulent_profiles_uncorrelated, axis=0), heights, color='darkred', linewidth=2.5, label='Mean')
axs[0].set_title("Uncorrelated", fontsize=24)
axs[0].set_xlabel("Wind Speed [m/s]", fontsize=19)
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].legend(fontsize=17)

# Time-correlated
for prof in turbulent_profiles_timecorr:
    axs[1].plot(prof, heights, alpha=0.2, color='seagreen')
axs[1].plot(np.mean(turbulent_profiles_timecorr, axis=0), heights, color='darkred', linewidth=2.5, label='Mean')
axs[1].set_title("Time-correlated", fontsize=24)
axs[1].set_xlabel("Wind Speed [m/s]", fontsize=19)
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].legend(fontsize=17)

# Time- and height-correlated
for prof in turbulent_profiles_bothcorr:
    axs[2].plot(prof, heights, alpha=0.2, color='slategray')
axs[2].plot(np.mean(turbulent_profiles_bothcorr, axis=0), heights, color='darkred', linewidth=2.5, label='Mean')
axs[2].set_title("Time- and Height-correlated", fontsize=24)
axs[2].set_xlabel("Wind Speed [m/s]", fontsize=19)
axs[2].grid(True, linestyle='--', alpha=0.5)
axs[2].legend(fontsize=17)

# Y-akse og ticks
for ax in axs:
    ax.set_ylabel("Height [m]", fontsize=19)
    ax.tick_params(labelsize=17)

plt.tight_layout()
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/turbulence_model.png', dpi = 300)
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



#%% Vindhastighet for hver bladseksjon og hvert tidspunkt

# Bruker tidsbasert rotasjon (ikke 360 samples, men ett per sekund)
num_timesteps = len(time_vector)  # Antall tidspunkter i én rotasjon

# Total bladspenn
blade_span_total = np.append(blade_span, 120)  # Extend to tip
num_blade_sections = len(blade_span_total)

# Høyde på hver seksjon (fra hub og utover)
blade_heights_sections = np.array(blade_span_total)

# Initialiser høyde- og vindmatriser
blade_section_rotation = np.zeros((num_blade_sections, num_timesteps))
wind_speed_rotation = np.zeros((num_blade_sections, num_timesteps))

# Loop gjennom hver seksjon på bladet
for section in range(num_blade_sections):
    # Høyde på seksjonen gjennom rotasjon (bruker tid-vinkel fra theta_deg)
    blade_section_rotation[section, :] = (
        hub_height + blade_heights_sections[section] * np.cos(np.radians(theta_deg))
    )

    # Hent vindprofilen for hvert tidspunkt fra listen u_turb_profiles
    for t in range(num_timesteps):
        wind_profile_t = turbulent_profiles[t]
        wind_speed_rotation[section, t] = np.interp(
            blade_section_rotation[section, t],
            heights,
            wind_profile_t
        )


plt.figure(figsize=(10, 5))

plt.plot(time_vector, wind_speed_rotation[-1, :], label='Tip of the blade', linestyle='-')
plt.plot(time_vector, wind_speed_rotation[0, :], label='Hub', linestyle='--')

plt.xlabel("Time [s]", fontsize=18)
plt.ylabel("Wind Speed [m/s]", fontsize=18)
plt.legend(fontsize=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.show()

#%%
# fig, ax1 = plt.subplots(figsize=(10, 5))

# # Hovedaksen: Rotasjonsvinkel (grader)
# ax1.plot(theta_deg, wind_speed_rotation[-1, :], label='Tip of the blade', linestyle='-')
# ax1.plot(theta_deg, wind_speed_rotation[0, :], label='Hub', linestyle='--')
# ax1.set_xlabel("Blade Position [degrees]", fontsize=18)
# ax1.set_ylabel("Wind Speed [m/s]", fontsize=18)
# ax1.legend(fontsize=15)
# ax1.grid(True, linestyle='--', alpha=0.7)
# ax1.tick_params(axis='both', which='major', labelsize=16)

# # Sekundær x-akse: Tid i sekunder
# ax2 = ax1.twiny()
# ax2.set_xlim(ax1.get_xlim())  # Matcher grenseverdier
# ax2.set_xticks(time_vector[::10])  # Velg færre ticks for oversiktlighet
# ax2.set_xticklabels(np.round(time_vector[::10], 1))  # Bruk tid i sekunder som labels
# ax2.set_xlabel("Time [s]", fontsize=16)
# ax2.tick_params(axis='x', labelsize=14)

# plt.tight_layout()
# plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/turbulence_rotation', dpi= 300)
# plt.show()


#%% relative velocity

blade_span_total_50 = blade_span_total[:50]  # Keep only the first 50 sections
num_blade_sections_50 = len(blade_span_total_50)
num_timesteps = len(time_vector)  # <- Viktig!

# Initialize arrays
phi_values_sections = np.zeros((num_blade_sections_50, num_timesteps))
w_sections = np.zeros((num_blade_sections_50, num_timesteps))
rot_speed_sections = np.zeros((num_blade_sections_50, num_timesteps))
omega_sections = np.zeros((num_blade_sections_50, num_timesteps))
u_sections = np.zeros((num_blade_sections_50, num_timesteps))
v_app_sections = np.zeros((num_blade_sections_50, num_timesteps))
a_sections = np.zeros((num_blade_sections_50, num_timesteps))

a_prime = 0  # tangential induction (still static)

# Beregn for hver bladseksjon og hvert tidspunkt
for section in range(num_blade_sections_50):
    # Midlere radius (r_central) for seksjonen
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2

    for i in range(num_timesteps):
        wind_speed_current = wind_speed_rotation[section, i]

        # Interpoler axial induction fra Ct-verdier
        a_section = get_axial_induction_factor(wind_speed_hub, wind_speeds_preformance, ct_values)
        a_sections[section, i] = a_section

        # Vind med aksial induksjon
        u_sections[section, i] = wind_speed_current * (1 - a_section)

        # Samme rotasjonshastighet og omega over hele bladet
        rot_speed_sections[section, i] = rot_speed_hub
        omega_sections[section, i] = omega_hub

        # Tangentiell hastighet med a'
        w_sections[section, i] = omega_hub * r_central * (1 + a_prime)

        # Inflow-vinkel
        phi_values_sections[section, i] = np.arctan(u_sections[section, i] / w_sections[section, i])

        # Apparent wind
        v_app_sections[section, i] = np.sqrt(u_sections[section, i]**2 + w_sections[section, i]**2)

# Tip speed og TSR
v_tip = omega_hub * rotor_radius
tsr = v_tip / wind_speed_hub
print("TSR:", tsr)


#%%
# Calculate the mean initial wind speed for each section (before axial induction)
mean_initial_wind_speed = np.mean(wind_speed_rotation, axis=1)

# Calculate the minimum and maximum initial wind speeds for each section
min_initial_wind_speed = np.min(wind_speed_rotation, axis=1)
max_initial_wind_speed = np.max(wind_speed_rotation, axis=1)

# # Define save path in the correct folder
# save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Modell_fremvisning/mean_initial_wind_speed.png"

# # Create figure
# plt.figure(figsize=(10, 5))

# # Plot mean wind speed with markers
# plt.plot(blade_span_total, mean_initial_wind_speed, marker='o', linestyle='-', color='blue', label='Mean Wind Speed')

# # Fill between min and max range
# plt.fill_between(blade_span_total, min_initial_wind_speed, max_initial_wind_speed, color='blue', alpha=0.2, label='Min-Max Range')

# # Set labels and title with larger fonts
# plt.xlabel('Blade Span (m)', fontsize=18)
# plt.ylabel('Wind Speed (m/s)', fontsize=18)
# plt.title('Wind Speed Along the Blade Span', fontsize=20)

# # Increase tick sizes
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

# # Show grid and legend
# plt.grid(True)
# plt.legend(fontsize=16, loc='lower left', frameon=True)

# # Save the figure with high resolution in the correct directory
# plt.savefig(save_path, dpi=300, bbox_inches='tight')

# # Show plot
# plt.show()


#%%
# Calculate the mean axial induction factor for each section along the blade
mean_axial_induction_factors = np.mean(a_sections, axis=1)


# Calculate the minimum and maximum axial induction factors for each section
min_axial_induction_factors = np.min(a_sections, axis=1)
max_axial_induction_factors = np.max(a_sections, axis=1)

        
#%%

# Calculate the mean effective wind speed after axial induction for each section
mean_effective_wind_speed = np.mean(u_sections, axis=1)

# Calculate the minimum and maximum effective wind speeds for each section
min_effective_wind_speed = np.min(u_sections, axis=1)
max_effective_wind_speed = np.max(u_sections, axis=1)





#%%
mean_v_app = np.mean(v_app_sections,axis=1)

# Calculate the minimum and maximum apparent wind speeds for each section
min_v_app = np.min(v_app_sections, axis=1)
max_v_app = np.max(v_app_sections, axis=1)


#%% blade pitch 
# Initialize a 2D array for blade pitch values at each section during the full rotation
blade_pitch_values_sections = np.zeros((num_blade_sections_50, len(time_vector)))

# Loop through each blade section
for section in range(num_blade_sections_50):
    # Loop through each tidspunkt i rotasjonen
    for i, _ in enumerate(time_vector):
        # Interpolate the blade pitch based on wind_speed_hub (same for all time steps)
        blade_pitch_values_sections[section, i] = np.interp(
            wind_speed_hub, wind_speeds_preformance, pitch_angles
        )


#%%
# Convert phi_values_sections to degrees
phi_values_sections_degrees = np.rad2deg(phi_values_sections)

# Calculate the average, minimum, and maximum of phi_values_sections in degrees
average_phi_sections = np.mean(phi_values_sections_degrees, axis=1)
min_phi_sections = np.min(phi_values_sections_degrees, axis=1)
max_phi_sections = np.max(phi_values_sections_degrees, axis=1)





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


#%% lift to drag
# Calculate the lift-to-drag ratio (Cl/Cd) for each section
lift_to_drag_ratio_sections = average_Cl_sections / average_Cd_sections

# Calculate the minimum and maximum lift-to-drag ratios for each section (optional)
min_lift_to_drag_ratio_sections = min_Cl_sections / max_Cd_sections
max_lift_to_drag_ratio_sections = max_Cl_sections / min_Cd_sections




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


#%% power coefficient

# Assuming `mean_axial_induction_factors` is already calculated as the average `a` for each section
# Calculate the power coefficient C_p for each section
power_coefficient_sections = 4 * mean_axial_induction_factors * (1 - mean_axial_induction_factors)**2

# Optionally calculate min and max C_p if you have min and max values for `a`
min_power_coefficient_sections = 4 * min_axial_induction_factors * (1 - min_axial_induction_factors)**2
max_power_coefficient_sections = 4 * max_axial_induction_factors * (1 - max_axial_induction_factors)**2



# Print the overall average power coefficient for the blade
overall_average_cp = np.mean(power_coefficient_sections)

#%%

cp_value = np.interp(wind_speed_hub, wind_speeds_preformance, cp_values)


# %% Lift and Drag Force Calculation

# Initialize arrays for lift and drag forces for each section
lift_force_sections = np.zeros_like(Cl_sections)
drag_force_sections = np.zeros_like(Cd_sections)
rho = 1.225  # Lufttetthet [kg/m^3]

num_timesteps = Cl_sections.shape[1]

# Loop over each blade section to calculate lift and drag forces
for section in range(num_blade_sections_50):
    # Calculate the blade segment length for each section
    if section < num_blade_sections_50 - 1:
        blade_segment = blade_span_total_50[section + 1] - blade_span_total_50[section]
    else:
        blade_segment = rotor_radius - blade_span_total_50[-1]  # Last segment to rotor radius

    # Get the chord length for the corresponding section
    chord_length = blade_chord[section]

    # Reference area A = chord length * blade segment
    area = chord_length * blade_segment

    # Loop through each time step (not blade positions anymore!)
    for i in range(num_timesteps):
        v_app = v_app_sections[section, i]  # Apparent wind speed
        Cl = Cl_sections[section, i]
        Cd = Cd_sections[section, i]

        # Lift and drag forces
        lift_force_sections[section, i] = 0.5 * rho * v_app**2 * Cl * area
        drag_force_sections[section, i] = 0.5 * rho * v_app**2 * Cd * area

# Sammenfattende statistikk
mean_lift_sections = np.mean(lift_force_sections, axis=1) / 1000  # [kN]
mean_drag_sections = np.mean(drag_force_sections, axis=1)         # [N]

min_lift_sections = np.min(lift_force_sections, axis=1) / 1000
max_lift_sections = np.max(lift_force_sections, axis=1) / 1000

min_drag_sections = np.min(drag_force_sections, axis=1)
max_drag_sections = np.max(drag_force_sections, axis=1)



# %% Total Lift and Drag (Oppdatert for tidssteg)

# Initialize variables for total lift and total drag
total_lift = 0
total_drag = 0

num_timesteps = lift_force_sections.shape[1]

# Loop through each blade section and calculate total lift and drag
for section in range(num_blade_sections_50):
    # Beregn gjennomsnittlig løft og drag for denne seksjonen over tid
    section_lift_avg = np.mean(lift_force_sections[section, :])
    section_drag_avg = np.mean(drag_force_sections[section, :])
    
    # Legg til i total løft og drag
    total_lift += section_lift_avg
    total_drag += section_drag_avg

# Konverter til kN
total_lift_kN = total_lift / 1000
total_drag_kN = total_drag / 1000


# %% Normal and Tangential Force Calculation
# Initialize arrays for normal and tangential forces
P_n_sections = np.zeros_like(lift_force_sections)
P_t_sections = np.zeros_like(lift_force_sections)

# Loop through each blade section to calculate the normal and tangential forces
for section in range(num_blade_sections_50):
    # Loop over hver tidsteg i stedet for rotasjonsvinkler
    for i in range(len(time_vector)):
        # Precompute sin and cos of the inflow angle (phi)
        cos_phi = np.cos(phi_values_sections[section, i])
        sin_phi = np.sin(phi_values_sections[section, i])

        # Normal force: Lift * cos(phi) + Drag * sin(phi)
        P_n_sections[section, i] = (
            lift_force_sections[section, i] * cos_phi +
            drag_force_sections[section, i] * sin_phi
        )

        # Tangential force: Lift * sin(phi) - Drag * cos(phi)
        P_t_sections[section, i] = (
            lift_force_sections[section, i] * sin_phi -
            drag_force_sections[section, i] * cos_phi
        )

# Konverter gjennomsnitt, min og maks til kN
mean_P_n_sections = np.mean(P_n_sections, axis=1) / 1000
min_P_n_sections = np.min(P_n_sections, axis=1) / 1000
max_P_n_sections = np.max(P_n_sections, axis=1) / 1000

mean_P_t_sections = np.mean(P_t_sections, axis=1) / 1000
min_P_t_sections = np.min(P_t_sections, axis=1) / 1000
max_P_t_sections = np.max(P_t_sections, axis=1) / 1000



#%% tip loss factor
# Initialize f and F arrays based on tidssteg
f = np.zeros((num_blade_sections_50, len(time_vector)))  # Prandtl's tip loss factor intermediate
F = np.zeros((num_blade_sections_50, len(time_vector)))  # Final tip loss factor

B = 3  # Number of blades, set to 3 for realistic turbiner

# Loop through sections
for section in range(num_blade_sections_50):
    # Calculate r_central for each section (average radius of the section)
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section

    # Loop through time steps instead of blade_positions
    for i in range(len(time_vector)):
        phi = phi_values_sections[section, i]  # already in radians

        # Unngå deling på null
        if np.sin(phi) == 0:
            f[section, i] = 0
            F[section, i] = 1  # No loss
        else:
            f[section, i] = (B / 2) * (rotor_radius - r_central) / (r_central * np.sin(phi))
            F[section, i] = (2 / np.pi) * np.arccos(np.exp(-f[section, i]))

# Calculate the average tip loss factor F for each section
average_F_sections = np.mean(F, axis=1)
min_F_sections = np.min(F, axis=1)
max_F_sections = np.max(F, axis=1)



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



#%%
# Initialize variables for total thrust and total torque
total_thrust = 0
total_torque = 0
B = 3  # Antall blader

# Loop through each blade section and calculate total thrust and torque
for section in range(num_blade_sections_50):
    section_thrust_sum = 0
    section_torque_sum = 0
    
    # Gjennomsnittsradius for seksjonen
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2

    # Loop over alle tidspunkter i rotasjonen
    for i in range(len(time_vector)):
        # Normal force bidrag til thrust
        thrust = P_n_sections[section, i] * B * F[section, i]
        section_thrust_sum += thrust
        
        # Tangential force bidrag til torque
        torque = P_t_sections[section, i] * B * r_central * F[section, i]
        section_torque_sum += torque
    
    # Snitt over rotasjonen
    section_thrust_avg = section_thrust_sum / len(time_vector)
    section_torque_avg = section_torque_sum / len(time_vector)
    
    # Akkumuler total thrust og torque
    total_thrust += section_thrust_avg
    total_torque += section_torque_avg

# Konverter til MN og MNm
total_thrust_MN = total_thrust / 1e6
total_torque_MNm = total_torque / 1e6




# %% Power Calculation

# omega_hub er allerede beregnet: (2 * np.pi * rot_speed_hub) / 60

# Initialize total power variable
total_power = 0

# Loop over each blade section and calculate total power using P = Q * omega
for section in range(num_blade_sections_50):
    section_torque_sum = 0

    # Gjennomsnittsradius for seksjonen
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2

    # Loop over tidspunktene i rotasjonen
    for i in range(len(time_vector)):
        # Torque = P_t * r * F
        torque = P_t_sections[section, i] * r_central * F[section, i]
        section_torque_sum += torque

    # Beregn snitt-torque for seksjonen
    section_torque_avg = section_torque_sum / len(time_vector)

    # Beregn effektbidrag fra seksjonen
    power_section = section_torque_avg * omega_hub  # P = Q * omega
    total_power += power_section

# Konverter til MW for én blad
total_power_mw = total_power / 1e6

# Skalér for tre blader
B = 3
total_power_mw_total = total_power_mw * B

# %% Power as function of time during one rotation

# # Initialiser array for effekt i hvert tidssteg
power_over_rotation = np.zeros(len(time_vector))

# Loop over alle tidspunkter i rotasjonen
for i in range(len(time_vector)):
    total_torque_at_i = 0
    
    for section in range(num_blade_sections_50):
        # Gjennomsnittsradius for seksjonen
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2

        # Beregn torque = P_t * r * F
        torque = P_t_sections[section, i] * r_central * F[section, i]
        total_torque_at_i += torque

    # P = Q * omega
    power_over_rotation[i] = total_torque_at_i * omega_hub

# # Konverter til MW og multipliser med antall blader
power_over_rotation_mw = power_over_rotation  / 1e6  #

# fig, ax1 = plt.subplots(figsize=(10, 5))

# # Primærakse: Rotasjonsvinkel
# rotation_degrees = (omega_hub * time_vector * 180 / np.pi) % 360
# ax1.plot(rotation_degrees, power_over_rotation_mw, label="Power output", color='navy')
# mean_power = np.mean(power_over_rotation_mw)
# ax1.axhline(mean_power, color='darkred', linestyle='--', label=f"Mean: {mean_power:.2f} MW")

# ax1.set_xlabel("Blade Position [°]", fontsize=18)
# ax1.set_ylabel("Power [MW]", fontsize=18)
# ax1.tick_params(axis='both', labelsize=15)
# ax1.grid(True, linestyle='--', alpha=0.7)

# # Sekundærakse: Tid
# ax2 = ax1.twiny()
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(rotation_degrees[::10])  # Færre ticks for klarhet
# ax2.set_xticklabels([f"{t:.1f}" for t in time_vector[::10]])
# ax2.set_xlabel("Time [s]", fontsize=18)
# ax2.tick_params(axis='x', labelsize=15)

# # Tittel og legende
# #plt.title("Power Output During One Blade Rotation", fontsize=16)
# ax1.legend(loc="upper center", fontsize=16)

# plt.tight_layout()
# plt.show()
#%%


# Plot kun mot tid
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time_vector, power_over_rotation_mw, label="Power output", color='navy')

# Gjennomsnittslinje
mean_power = np.mean(power_over_rotation_mw)
ax.axhline(mean_power, color='darkred', linestyle='--', label=f"Mean: {mean_power:.2f} MW")

# Aksetitler og formatering
ax.set_xlabel("Time [s]", fontsize=18)
ax.set_ylabel("Power [MW]", fontsize=18)
ax.tick_params(axis='both', labelsize=15)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=16)

plt.tight_layout()
plt.show()


#%%


#%%

# Initialize arrays to store power contributions for each section
average_power_sections = []
min_power_sections = []
max_power_sections = []

# Loop over each blade section to calculate the average, min, and max power contributions
for section in range(num_blade_sections_50):
    # List to store power values for all tidssteg
    power_values = []

    # r_central beregnes utenfor løkka
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section

    # Loop over tidspunkter i rotasjonen
    for i in range(len(time_vector)):
        # Torque
        torque = P_t_sections[section, i] * r_central * F[section, i]

        # Power for dette tidssteget (i MW)
        power_value = torque * omega_hub / 1e6
        power_values.append(power_value)

    # Stats for seksjonen
    average_power_sections.append(np.mean(power_values))
    min_power_sections.append(np.min(power_values))
    max_power_sections.append(np.max(power_values))



#%%

def print_turbine_report(
    wind_speed_hub, rot_speed_hub, v_tip, tsr, mean_initial_wind_speed, 
    average_axial_induction_factor, mean_effective_wind_speed, average_lift_to_drag_ratio,
    overall_average_cp, total_thrust_MN, total_torque_MNm, total_power_mw, total_power_mw_total,
    total_lift_kN, total_drag_kN, ct, bladepitch
):

    print("--------------------------------")
    print(f"Wind Speed at Hub: {wind_speed_hub:.2f} m/s")
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


#%% Bending moment


# File path
file_path_blade_data = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/IEA-15-240-RWT_ElastoDyn_blade.dat"

# Total blademass (kg)
total_blade_mass = 65250  
blade_length = 120  # meter

# read file
with open(file_path_blade_data, "r") as file:
    lines = file.readlines()

# start and end value
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

# DataFrame
blade_data = pd.read_csv(
    file_path_blade_data,
    skiprows=start_index,
    nrows=end_index - start_index,
    sep=r"\s+",
    header=None,
    names=["BlFract", "PitchAxis", "StrcTwst", "BMassDen", "FlpStff", "EdgStff"],
    engine="python",
)

# calculating length of each section
blade_data["Section Length"] = np.diff(blade_data["BlFract"], prepend=0) * blade_length

# density and lenght
section_densities = blade_data["BMassDen"].values  # kg/m
section_lengths = blade_data["Section Length"].values  # meter

# calculating mass per section based on total blade mass and the length and density of each section
mass_per_section = (section_lengths * section_densities) / np.sum(section_lengths * section_densities) * total_blade_mass

# adding to DataFrame
blade_data["Section Mass"] = mass_per_section


#%% FLAPWISE

# Initialize thrust (flapwise) moment array
aerodynamic_force = np.zeros((num_blade_sections_50, len(time_vector)))

for section in range(num_blade_sections_50):
    # Mid-radius for the section
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2

    # Retrieve twist for the section (degrees → radians)
    beta_twist = np.radians(blade_twist[section])

    for i in range(len(time_vector)):
        # Retrieve aerodynamic quantities
        F_n = P_n_sections[section, i]
        F_t = P_t_sections[section, i]
        phi = phi_values_sections[section, i]
        theta_pitch = np.radians(blade_pitch_values_sections[section, i])
        F_tip = F[section, i]

        # Effective inflow angle
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
gravitational_moments_sections = np.zeros((len(blade_data), len(time_vector)))
    
for section in range(len(blade_data)):
    # Mid-radius for the section (average radius)
    if section < len(blade_data) - 1:
        r_central = (blade_data["BlFract"].iloc[section] + blade_data["BlFract"].iloc[section + 1]) / 2 * blade_length
    else:
        r_central = blade_length  # Last section
            
    # Mass of the section
    m_i = blade_data["Section Mass"].iloc[section]
        
    # The gravitational moment is constant for all positions, but we iterate for structural consistency
    for i in range(len(time_vector)):
        # Hent faktisk vinkel (0–360°)
        theta = theta_deg[i]

        # Gravitational moment varies with blade position in rotation
        gravitational_moment = m_i * g * r_central * np.sin(np.radians(tilt_angle - cone_angle * np.cos(np.radians(theta))))

        gravitational_moments_sections[section, i] = gravitational_moment

# Total flapwise moment = aerodynamic + gravitational contributions
total_flapwise = aerodynamic_force + gravitational_moments_sections


#%%
# Initialize array for cumulative moments (same dimensions as thrust_moment_sections)
rotor_aerodynamic = np.zeros((num_blade_sections_50, len(time_vector)))

# Iterate backwards to compute cumulative moments
for section in range(num_blade_sections_50 - 1, -1, -1):  # From last to first section
    if section == num_blade_sections_50 - 1:  # Last section (outermost part of the blade)
        rotor_aerodynamic[section] = aerodynamic_force[section]
    else:
        rotor_aerodynamic[section] = (
            aerodynamic_force[section] + rotor_aerodynamic[section + 1]
        )

rotor_gravitational = np.zeros((len(blade_data), len(time_vector)))

# Iterate backwards to compute cumulative gravitational moments
for section in range(len(blade_data) - 1, -1, -1):
    if section == len(blade_data) - 1:  # Last section (outermost part of the blade)
        rotor_gravitational[section] = gravitational_moments_sections[section]
    else:
        rotor_gravitational[section] = (
            gravitational_moments_sections[section] + rotor_gravitational[section + 1]
        )
        
rotor_total_flap = np.zeros((len(blade_data), len(time_vector)))

# Iterate backwards to compute cumulative total flapwise moments
for section in range(len(blade_data) - 1, -1, -1):
    if section == len(blade_data) - 1:  # Last section (outermost part of the blade)
        rotor_total_flap[section] = total_flapwise[section]
    else:
        rotor_total_flap[section] = (
            total_flapwise[section] + rotor_total_flap[section + 1]
        )     

# Plotting
plt.figure(figsize=(14, 5))

plt.plot(time_vector, rotor_gravitational[0] * 1e-6, label="Gravitational", linestyle='--')
plt.plot(time_vector, rotor_aerodynamic[0] * 1e-6, label="Aerodynamic Forces", linestyle='-.')
plt.plot(time_vector, rotor_total_flap[0] * 1e-6, label="Total", linestyle='-')

# Plot settings
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("(MNm)", fontsize=18)
plt.title(f"Flapwise Bending Moment", fontsize=20)
plt.legend(
    fontsize=17,
    title="Bending Moment Components",
    title_fontsize=18,
    loc='upper left',
    bbox_to_anchor=(1.02, 1)  # Moves the legend outside the plot
)

plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()

# Show plot
plt.show()


#%% EDGEWISE
adjusted_theta_deg = (theta_deg + 270) % 360  # array med lengde = len(time_vector)

# Initialize array for edgewise aerodynamic bending moment (excluding gravity!)
edgewise_moments_sections = np.zeros((num_blade_sections_50, len(time_vector)))

# Compute edgewise bending moment for each section
for section in range(num_blade_sections_50):
    # Mid-radius for the section
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2

    # Twist for the section (degrees → radians)
    beta_twist = np.radians(blade_twist[section])

    for i in range(len(time_vector)):
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
gravitational_moments_sections_edge = np.zeros((num_blade_sections_50, len(time_vector)))

# Compute gravitational moment for each section
for section in range(num_blade_sections_50):
    # Mid-radius for the section (average radius)
    if section < num_blade_sections_50 - 1:
        r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
    else:
        r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section

    # Mass of the section
    m_i = blade_data["Section Mass"].iloc[section]

    for i in range(len(time_vector)):
        blade_angle = adjusted_theta_deg[i]  # 0° = på vei ned
        gravitational_moment = m_i * g * r_central * np.cos(np.radians(blade_angle))
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

plt.plot(time_vector, rotor_gravity_edge[0] * 1e-6, label="Gravitational", linestyle='--', color='green')
plt.plot(time_vector, rotor_aerodynamic_edge[0] * 1e-6, label="Aerodynamic Forces", linestyle='-', color='blue')
plt.plot(time_vector, cumulative_total_edgewise_moments[0] * 1e-6, label="Total", linestyle='-.', color='red')

# Plot settings
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("(MNm)", fontsize=18)
plt.title(f"Edgewise Bending Moment", fontsize=20)
plt.legend(fontsize=17, title="Bending Moment Components", title_fontsize=18)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()

# Show plot
plt.show()


#%%
import os
import pandas as pd

# nr = 100

# # === MAPPENE ===
# power_dir = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/Power_2"
# flap_dir = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/Flap"
# edge_dir = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/Edge"

# # Lag mapper hvis de ikke finnes
# os.makedirs(power_dir, exist_ok=True)
# os.makedirs(flap_dir, exist_ok=True)
# os.makedirs(edge_dir, exist_ok=True)

# # === 1. POWER ===
# df_power = pd.DataFrame({
#     "Time [s]": time_vector,
#     "Blade Position [deg]": rotation_degrees,
#     "Power [MW]": power_over_rotation_mw
# })
# df_power.to_csv(os.path.join(power_dir, f"power_output_rotation_{nr}.csv"), index=False)

# # === 2. FLAPWISE MOMENT ===
# df_flap = pd.DataFrame({
#     "Time [s]": time_vector,
#     "Blade Position [deg]": rotation_degrees,
#     "Flapwise Moment [Nm]": rotor_total_flap[0]
# })
# df_flap.to_csv(os.path.join(flap_dir, f"flap_output_rotation_{nr}.csv"), index=False)

# # === 3. EDGEWISE MOMENT ===
# df_edge = pd.DataFrame({
#     "Time [s]": time_vector,
#     "Blade Position [deg]": rotation_degrees,
#     "Edgewise Moment [Nm]": cumulative_total_edgewise_moments[0]
# })
# df_edge.to_csv(os.path.join(edge_dir, f"edge_output_rotation_{nr}.csv"), index=False)

# print("Filer lagret:")
# print(f"Power → {power_dir}")
# print(f"Flap  → {flap_dir}")
# print(f"Edge  → {edge_dir}")


#%%
import glob

# Sti til mappen med data
folder_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/Power_2"

# Finn alle relevante filer
csv_files = sorted(glob.glob(os.path.join(folder_path, "power_output_rotation_*.csv")))

# Initialiser tom liste for å lagre data
all_power_data = []

# Les inn alle filer
for file in csv_files:
    df = pd.read_csv(file)
    all_power_data.append(df['Power [MW]'])

# Lag en array med samme tid (samme for alle filer)
time = pd.read_csv(csv_files[0])['Time [s]']

# Konverter til DataFrame
power_df = pd.DataFrame(all_power_data).T  # Transponér slik at hver kolonne er én rotasjon

# Beregn gjennomsnitt
mean_power_rotation = power_df.mean(axis=1)

print(len(csv_files))
mean_power = np.mean(power_df)
print(mean_power*3)
# Plot
plt.figure(figsize=(12, 6))

# Plott alle simuleringer
for i in range(power_df.shape[1]):
    plt.plot(time, power_df.iloc[:, i], alpha=0.5, color = 'steelblue')

# Plott gjennomsnitt
plt.axhline(mean_power, color='darkred', linestyle='--', linewidth=2.5,
            label=f'Mean Power Output: {mean_power:.2f} MW')



plt.xlabel("Time [s]", fontsize=18)
plt.ylabel("Power [MW]", fontsize=18)
plt.title(f"Power Output Over {len(csv_files)} Turbulent Rotations", fontsize=22)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=17, loc='upper center')
plt.xticks(fontsize=16)  # større x-ticks
plt.yticks(fontsize=16)  # større y-ticks
plt.tight_layout()
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/power_turbulence.pdf', dpi =300)
plt.show()




#%% FLAPWISE PLOT

import glob
import pandas as pd
import matplotlib.pyplot as plt

# === Innlesing ===
flap_folder = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/Flap"
flap_files = sorted(glob.glob(os.path.join(flap_folder, "flap_output_rotation_*.csv")))

flap_data = []
for file in flap_files:
    df = pd.read_csv(file)
    flap_data.append(df["Flapwise Moment [Nm]"])

time = pd.read_csv(flap_files[0])["Time [s]"]
rotation_degrees = pd.read_csv(flap_files[0])["Blade Position [deg]"]

flap_df = pd.DataFrame(flap_data).T
mean_flap = flap_df.mean(axis=1)

# === Plot ===
plt.figure(figsize=(12, 6))

for i in range(flap_df.shape[1]):
    plt.plot(rotation_degrees, flap_df.iloc[:, i] * 1e-6, alpha=0.5, color='teal')

plt.plot(rotation_degrees, mean_flap * 1e-6, color='darkred', linewidth=2.5, label="Mean")

plt.xlabel("Blade Position [°]", fontsize=18)
plt.ylabel("Flapwise Moment [MNm]", fontsize=18)
plt.title(f"Flapwise Bending Moment over {len(flap_files)} Rotations", fontsize=22)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=17, loc='upper center')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig("/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/flapwise_turbulence.pdf", dpi=300)
plt.show()

#%%
# FLAPWISE with standard deviation shading
plt.figure(figsize=(12, 6))

# Plot all 100 simuleringer (som før)
# for i in range(flap_df.shape[1]):
#     plt.plot(rotation_degrees, flap_df.iloc[:, i] * 1e-6, alpha=0.3, color='teal')

# Beregn standardavvik
std_flap = flap_df.std(axis=1)

# Plot gjennomsnitt og ±1 standardavvik
plt.plot(rotation_degrees, mean_flap * 1e-6, color='darkred', linewidth=2.5, label="Mean")
plt.fill_between(
    rotation_degrees,
    (mean_flap - std_flap) * 1e-6,
    (mean_flap + std_flap) * 1e-6,
    color='darkred',
    alpha=0.2,
    label="±1 Std Dev"
)

plt.xlabel("Blade Position [°]", fontsize=18)
plt.ylabel("Flapwise Moment [MNm]", fontsize=18)
plt.title(f"Flapwise Bending Moment over {len(flap_files)} Rotations", fontsize=22)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=17, loc='upper center')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig("/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/flapwise_turbulence_std.pdf", dpi=300)
plt.show()


#%% EDGEWISE PLOT

edge_folder = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/Edge"
edge_files = sorted(glob.glob(os.path.join(edge_folder, "edge_output_rotation_*.csv")))

edge_data = []
for file in edge_files:
    df = pd.read_csv(file)
    edge_data.append(df["Edgewise Moment [Nm]"])

rotation_degrees = pd.read_csv(edge_files[0])["Blade Position [deg]"]

edge_df = pd.DataFrame(edge_data).T
mean_edge = edge_df.mean(axis=1)

plt.figure(figsize=(12, 6))

for i in range(edge_df.shape[1]):
    plt.plot(rotation_degrees, edge_df.iloc[:, i] * 1e-6, alpha=0.5, color='slateblue')

plt.plot(rotation_degrees, mean_edge * 1e-6, color='darkred', linewidth=2.5, label="Mean")

plt.xlabel("Blade Position [°]", fontsize=18)
plt.ylabel("Edgewise Moment [MNm]", fontsize=18)
plt.title(f"Edgewise Bending Moment over {len(edge_files)} Rotations", fontsize=22)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=17, loc='upper center')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig("/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/edgewise_turbulence.pdf", dpi=300)
plt.show()

#%%
# EDGEWISE with standard deviation shading
plt.figure(figsize=(12, 6))

# # Plot all 100 simuleringer
# for i in range(edge_df.shape[1]):
#     plt.plot(rotation_degrees, edge_df.iloc[:, i] * 1e-6, alpha=0.3, color='slateblue')

# Beregn standardavvik
std_edge = edge_df.std(axis=1)

# Plot gjennomsnitt og ±1 standardavvik
plt.plot(rotation_degrees, mean_edge * 1e-6, color='darkred', linewidth=2.5, label="Mean")
plt.fill_between(
    rotation_degrees,
    (mean_edge - std_edge) * 1e-6,
    (mean_edge + std_edge) * 1e-6,
    color='darkred',
    alpha=0.2,
    label="±1 Std Dev"
)

plt.xlabel("Blade Position [°]", fontsize=18)
plt.ylabel("Edgewise Moment [MNm]", fontsize=18)
plt.title(f"Edgewise Bending Moment over {len(edge_files)} Rotations", fontsize=22)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=17, loc='upper center')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig("/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/turbulence/edgewise_turbulence_std.pdf", dpi=300)
plt.show()

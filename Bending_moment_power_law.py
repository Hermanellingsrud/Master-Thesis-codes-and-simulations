#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:39:58 2024

@author: hermanellingsrud
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# Vindhastigheter i området 0–25 m/s
wind_speeds = np.arange(0, 25.5, 0.5)
heights = np.array([10, 20, 50, 100, 150, 200, 250, 300])  # Høyder opp til 300m
alpha = 0.14  # Vindskjæringseksponent, bytt til 0.00 for homogent baseline
hub_height = 150  

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

rotor_diameter = 240  # Rotor diameter (i meter)
rotor_radius = rotor_diameter / 2  # Rotor radius (halvparten av diameteren, i meter)

# Høyder for bladets topp- og bunnposisjon basert på rotorens diameter og navhøyde
H_top = hub_height + rotor_radius  # Høyde på bladspissen øverst (270 m)
H_bottom = hub_height - rotor_radius  # Høyde på bladspissen nederst (30 m)

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


# Loop over hver vindhastighet fra 0 til 25 m/s
for wind_speed in wind_speeds:
    # Sett navhastighet og lagre for denne vinden
    wind_speed_hub = wind_speed
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

#%% Wind for each section with wind shear
blade_span_total = np.append(blade_span, 120)  # Extend blade span to the full rotor radius (120 meters)
num_blade_sections = len(blade_span_total)

# Heights for each section along the blade, based on span points
blade_heights_sections = np.array(blade_span_total)

# Dictionary to store wind speed rotation results for each wind speed
wind_speed_rotation_all = {}

# Loop over each wind speed from 0 to 25 m/s
for wind_speed in wind_speeds:
    # Calculate wind shear profile based on hub height wind speed
    wind_speeds_current = wind_speed * (heights / hub_height) ** alpha  # Wind shear profile at different heights

    # Initialize arrays for blade section heights and wind speeds during rotation
    blade_section_rotation = np.zeros((num_blade_sections, len(blade_positions)))  # Height of each section during rotation
    wind_speed_rotation = np.zeros((num_blade_sections, len(blade_positions)))     # Wind speed at each section during rotation

    # Loop over each blade section to calculate height and interpolated wind speed during rotation
    for section in range(num_blade_sections):
        # Calculate the height of each section during rotation
        blade_section_rotation[section, :] = hub_height + blade_heights_sections[section] * np.cos(np.radians(blade_positions))

        # Interpolate the wind speed at each section height during rotation based on wind shear profile
        wind_speed_rotation[section, :] = np.interp(blade_section_rotation[section, :], heights, wind_speeds_current)

    # Store the results for this wind speed in the dictionary
    wind_speed_rotation_all[wind_speed] = wind_speed_rotation


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



for wind_speed in wind_speeds:
    # Hent rotasjonshastigheter og beregn omega
    wind_speed_hub = wind_speed
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
for wind_speed in wind_speeds:
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

    # Loop through each blade
    for blade_offset in [0, 120, 240]:  # Blade positions (0°, 120°, 240°)
        # Loop through each blade section
        for section in range(num_blade_sections_50):
            # Sum thrust and torque contributions over all rotational positions for this section
            section_thrust_sum = 0
            section_torque_sum = 0
            
            # Loop over each rotational position
            for i, blade_position in enumerate(blade_positions):
                # Adjust blade position for the current blade
                adjusted_position = (blade_position + blade_offset) % 360

                # Find the nearest rotor index for this adjusted position
                rotor_index = (np.abs(blade_positions - adjusted_position)).argmin()

                # Thrust at this rotational position
                thrust = P_n_sections[section, rotor_index] * F_sections[section, rotor_index]
                section_thrust_sum += thrust

                # Calculate r_central for each section (average radius of the section)
                if section < num_blade_sections_50 - 1:
                    r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
                else:
                    r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section

                # Torque at this rotational position
                torque = P_t_sections[section, rotor_index] * r_central * F_sections[section, rotor_index]
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

# Print results for each wind speed
for wind_speed in wind_speeds:
    print(
        f"Wind Speed: {wind_speed} m/s, "
        f"Total Thrust: {total_thrust_all[wind_speed]:.6f} MN, "
        f"Total Torque: {total_torque_all[wind_speed]:.6f} MN·m"
    )
#%%

# Spesifiser vindhastighet for plott
wind_speed_to_plot = 15  # Velg en vindhastighet for plottet

# Initialiser arrays for thrust og torque gjennom hele rotasjonen
total_thrust_rotation = np.zeros(len(blade_positions))
total_torque_rotation = np.zeros(len(blade_positions))

# Beregn total thrust og torque gjennom rotasjonen
if wind_speed_to_plot in P_n_all and wind_speed_to_plot in P_t_all:
    P_n_sections = P_n_all[wind_speed_to_plot]
    P_t_sections = P_t_all[wind_speed_to_plot]
    F_sections = tip_loss_factor_all[wind_speed_to_plot]

    # Loop gjennom alle rotorposisjoner
    for i, blade_position in enumerate(blade_positions):
        thrust_sum = 0
        torque_sum = 0

        # Bidrag fra alle blader
        for blade_offset in [0, 120, 240]:  # Blade positions (0°, 120°, 240°)
            adjusted_position = (blade_position + blade_offset) % 360
            rotor_index = (np.abs(blade_positions - adjusted_position)).argmin()

            # Loop gjennom alle bladseksjoner
            for section in range(num_blade_sections_50):
                # Beregn thrust og torque for denne posisjonen
                r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2 if section < num_blade_sections_50 - 1 else (rotor_radius + blade_span_total_50[-1]) / 2
                thrust = P_n_sections[section, rotor_index] * F_sections[section, rotor_index]
                torque = P_t_sections[section, rotor_index] * r_central * F_sections[section, rotor_index]

                # Legg til bidrag for denne seksjonen
                thrust_sum += thrust
                torque_sum += torque

        # Lagre total thrust og torque for denne rotorposisjonen
        total_thrust_rotation[i] = thrust_sum / 1e6  # Konverter til MN
        total_torque_rotation[i] = torque_sum / 1e6  # Konverter til MN·m



    
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



#%% Bøyemoment

bending_moments_all = {}

for wind_speed in wind_speeds:
    # Retrieve necessary aerodynamic data
    P_n_sections = P_n_all[wind_speed]
    P_t_sections = P_t_all[wind_speed]  # Tangential forces
    phi_sections = phi_values_all[wind_speed]  # inflow angle (radians)
    F_sections = tip_loss_factor_all[wind_speed]
    pitch_sections = blade_pitch_values_all[wind_speed]  # shape: (sections, positions)

    aerodynamic_flap_sections = np.zeros_like(P_n_sections)

    for section in range(num_blade_sections_50):
        # Mid-radius of the section
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2

        # Get twist for this section (in radians)
        beta_twist = np.radians(blade_twist[section])

        for i in range(len(blade_positions)):
            phi = phi_sections[section, i]                    # inflow angle in radians
            theta_pitch = np.radians(pitch_sections[section, i])  # pitch per section and position

            F_n = P_n_sections[section, i]
            F_t = P_t_sections[section, i]
            F = F_sections[section, i]

            # Compute local flapwise force
            phi_eff = phi - (theta_pitch + beta_twist)
            flapwise_local = F_n * np.cos(phi_eff) - F_t * np.sin(phi_eff)

            # Bending moment = local force * r * tip loss factor
            aerodynamic_flap_sections[section, i] = flapwise_local * r_central * F

    bending_moments_all[wind_speed] = aerodynamic_flap_sections





#%%
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

print(blade_data)

print(np.sum(blade_data["Section Mass"]))




#%% flap gravitasjon
# Konstant for gravitasjonsakselerasjon
g = 9.81  # m/s^2

# Tiltvinkel
tilt_angle = 6  # grader

cone_angle = 4

# Initialisere dictionary for gravitasjonsmoment
gravitational_moments_all = {}

# Beregning av gravitasjonsmoment for alle seksjoner og posisjoner
for wind_speed in wind_speeds:
    # Initialiser en 2D-array for seksjoner og rotorposisjoner
    gravitational_moments_sections = np.zeros((len(blade_data), len(blade_positions)))
    
    for section in range(len(blade_data)):
        # Midtradius for seksjonen (gjennomsnittlig radius)
        if section < len(blade_data) - 1:
            r_central = (blade_data["BlFract"].iloc[section] + blade_data["BlFract"].iloc[section + 1]) / 2 * blade_length
        else:
            r_central = blade_length  # Siste seksjon
            
        
        # Massen for seksjonen
        m_i = blade_data["Section Mass"].iloc[section]
        
        # Gravitasjonsmomentet er konstant for alle posisjoner, men vi itererer for strukturens skyld
        for i in range(len(blade_positions)):
            # Beregn gravitasjonsmomentet for denne posisjonen
            gravitational_moment = m_i * g * r_central * np.sin(np.radians(tilt_angle-cone_angle*np.cos(np.radians(i))))
            
            # Lagre verdien i arrayen
            gravitational_moments_sections[section, i] = gravitational_moment

    # Lagre gravitasjonsmomentene for denne vindhastigheten
    gravitational_moments_all[wind_speed] = gravitational_moments_sections







#%%
# Initialisere dictionary for totalt flapwise bøyemoment
total_flapwise_moments_all = {}

for wind_speed in wind_speeds:
    if wind_speed in bending_moments_all and wind_speed in gravitational_moments_all:
        # Hent aerodynamisk og gravitasjonsmoment
        aero_moments = bending_moments_all[wind_speed]
        grav_moments = gravitational_moments_all[wind_speed]
        
        # Totalt moment som en sum av aerodynamisk og gravitasjonsmoment
        total_moments = aero_moments + grav_moments
        
        # Lagre resultatene
        total_flapwise_moments_all[wind_speed] = total_moments
        

#%%
# Beregning av kumulative aerodynamiske momenter
rotor_flap_all = {}
for wind_speed in wind_speeds:
    # Hent aerodynamiske bøyemomenter for gjeldende vindhastighet
    aero_moments = bending_moments_all[wind_speed]
    rotor_moments_sections = np.zeros_like(aero_moments)

    # Iterer baklengs for å beregne kumulative momenter
    for section in range(num_blade_sections_50 - 1, -1, -1):  # Fra siste til første seksjon
        if section == num_blade_sections_50 - 1:  # Siste seksjon
            rotor_moments_sections[section] = aero_moments[section]
        else:
            rotor_moments_sections[section] = (
                aero_moments[section] + rotor_moments_sections[section + 1]
            )
    
    # Lagre kumulative aerodynamiske momenter
    rotor_flap_all[wind_speed] = rotor_moments_sections

# Beregning av kumulative gravitasjonsmomenter
rotor_flap_gravity_all = {}
for wind_speed in gravitational_moments_all:
    gravitational_moments = gravitational_moments_all[wind_speed]
    rotor_flap_grav = np.zeros_like(gravitational_moments)

    # Iterer baklengs for å beregne kumulative gravitasjonsmomenter
    for section in range(len(blade_data) - 1, -1, -1):  # Fra siste til første seksjon
        if section == len(blade_data) - 1:  # Siste seksjon
            rotor_flap_grav[section, :] = gravitational_moments[section, :]
        else:
            rotor_flap_grav[section, :] = (
                gravitational_moments[section, :] + rotor_flap_grav[section + 1, :]
            )
    
    # Lagre kumulative gravitasjonsmomenter
    rotor_flap_gravity_all[wind_speed] = rotor_flap_grav
# Kombinere kumulative momenter for total flapwise moment
rotor_flap_total_all = {}
for wind_speed in wind_speeds:
    if wind_speed in rotor_flap_all and wind_speed in rotor_flap_gravity_all:
        # Hent kumulative aerodynamiske og gravitasjonsmomenter
        aero_rotor = rotor_flap_all[wind_speed]
        grav_rotor = rotor_flap_gravity_all[wind_speed]
        
        # Totalt kumulativt moment
        rotor_flap_total_moments = aero_rotor + grav_rotor
        
        # Lagre totalt kumulativt moment
        rotor_flap_total_all[wind_speed] = rotor_flap_total_moments

wind_speed_to_plot = 15   
# Plot kumulative momenter for flapwise gjennom en hel rotasjon
plt.figure(figsize=(10, 6))

plt.plot(blade_positions, rotor_flap_gravity_all[wind_speed_to_plot][0] * 1e-6, label="Gravitational", linestyle='--')
plt.plot(blade_positions, rotor_flap_all[wind_speed_to_plot][0] * 1e-6, label=f"Aerodynamic Forces", linestyle='-.')
plt.plot(blade_positions, rotor_flap_total_all[wind_speed_to_plot][0] * 1e-6, label="Total", linestyle='-')

# Plotinnstillinger
plt.xlabel("Blade Position (degrees)", fontsize=18)
plt.ylabel("(MNm)", fontsize=18)
plt.title(f"Flapwise Bending Moments Over One Full Rotation {wind_speed_to_plot} m/s", fontsize=20)
plt.legend(fontsize=17, title="Bending Moment Components", title_fontsize=18)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.savefig("/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Bøymoment/flapwise_moment_components_full_rotation.png", dpi=300)

plt.show()

        


#%%

# --- 1. Les inn baseline-data og konverter enheter ---
baseline_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Baseline/baseline_flapwise.csv'
baseline_df = pd.read_csv(baseline_path)

# Del på 1e6 for å få MNm
baseline_df['Average Flapwise Bending Moment (MNm)'] /= 1e6

baseline_wind_speeds = baseline_df['Wind Speed (m/s)'].values
baseline_moments = baseline_df['Average Flapwise Bending Moment (MNm)'].values

# --- 2. Modellens flapwise moments ---
average_moments_per_wind_speed = []

for wind_speed, total_moments in total_flapwise_moments_all.items():
    avg_moment = np.sum(np.mean(total_moments, axis=1))  # summert gjennomsnitt per seksjon
    average_moments_per_wind_speed.append((wind_speed, avg_moment))

average_moments_per_wind_speed = np.array(average_moments_per_wind_speed)
model_wind_speeds = average_moments_per_wind_speed[:, 0]
model_moments = average_moments_per_wind_speed[:, 1] * 1e-6  # Nm → MNm

# --- 3. Plot sammenligning ---
plt.figure(figsize=(10, 5))
plt.plot(model_wind_speeds, model_moments, marker='o', linestyle='-', label=f'Alpha = {alpha}')
plt.plot(baseline_wind_speeds, baseline_moments, marker='x', linestyle='--', label='Homogeneous baseline')

plt.xlabel("Wind Speed (m/s)",fontsize = 18)
plt.ylabel("(MNm)", fontsize = 18)
plt.title("Average Flapwise Bending Moment", fontsize = 20)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
plt.xlim(3, 25)
#plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/average_flap.pdf', dpi = 300)
plt.show()

#%%


# --- 3. Plot sammenligning ---
plt.figure(figsize=(10, 5))
plt.plot(model_wind_speeds, model_moments/baseline_moments, marker='o', linestyle='-', label=f'Alpha = {alpha}')
plt.plot(baseline_wind_speeds, baseline_moments/baseline_moments, marker='x', linestyle='--', label='Homogeneous baseline')

plt.xlabel("Wind Speed (m/s)",fontsize = 18)
plt.ylabel("", fontsize = 18)
plt.title("Normalized Flapwise Bending Moment", fontsize = 20)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
plt.xlim(3, 25)
#plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/normalized_average_flap.pdf', dpi = 300)
plt.show()



# Lag DataFrame med normaliserte verdier
normalized_flap_df = pd.DataFrame({
    'Wind Speed (m/s)': model_wind_speeds,
    'Normalized Flapwise Moment (-)': model_moments / baseline_moments
})

# Filsti for lagring
save_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Baseline/Alfa_verdier/Flap/normalized_flapwise_alpha_{alpha}.csv'.format(alpha)

# Lagre til CSV
normalized_flap_df.to_csv(save_path, index=False)
print(f"Saved to: {save_path}")

#%% edgewise

# Forskyt rotorposisjonene slik at 0° er toppen av rotasjonen
adjusted_blade_positions = (blade_positions + 270) % 360



edgewise_moments_all = {}

for wind_speed in wind_speeds:
    # Hent nødvendige aerodynamiske data
    P_n_sections = P_n_all[wind_speed]
    P_t_sections = P_t_all[wind_speed]
    phi_sections = phi_values_all[wind_speed]
    F_sections = tip_loss_factor_all[wind_speed]
    pitch_sections = blade_pitch_values_all[wind_speed]

    edgewise_moments_sections = np.zeros_like(P_n_sections)

    for section in range(num_blade_sections_50):
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2

        m_i = blade_data["Section Mass"].iloc[section]
        beta_twist = np.radians(blade_twist[section])

        for i, blade_position in enumerate(adjusted_blade_positions):
            phi = phi_sections[section, i]
            theta_pitch = np.radians(pitch_sections[section, i])

            F_n = P_n_sections[section, i]
            F_t = P_t_sections[section, i]
            F = F_sections[section, i]

            # Beregn lokal edgewise kraft
            phi_eff = phi - (theta_pitch + beta_twist)
            edgewise_local = F_n * np.sin(phi_eff) + F_t * np.cos(phi_eff)

            # Aerodynamisk edgewise moment
            aerodynamic_moment = edgewise_local * r_central * F

            # Gravitasjonsmoment
            gravitational_moment = m_i * g * r_central * np.cos(np.radians(blade_position))

            # Totalt edgewise moment
            edgewise_moments_sections[section, i] = aerodynamic_moment + gravitational_moment

    edgewise_moments_all[wind_speed] = edgewise_moments_sections





#%%
# Initialiser dictionary for kumulative torque-momenter (kun aerodynamikk)
rotor_aerodynamic_edge_moments_all = {}

# Beregning av kumulative torque-momenter for alle vindhastigheter
for wind_speed in wind_speeds:
    # Hent nødvendige data
    P_n_sections = P_n_all[wind_speed]  # Normal forces
    P_t_sections = P_t_all[wind_speed]  # Tangential forces
    phi_sections = phi_values_all[wind_speed]  # Inflow angles
    F_sections = tip_loss_factor_all[wind_speed]  # Tip loss factors
    pitch_sections = blade_pitch_values_all[wind_speed]  # Pitch per section

    # Initialiser array for kumulative torque-momenter
    torque_sections = np.zeros_like(P_n_sections)

    for section in range(num_blade_sections_50):
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2

        beta_twist = np.radians(blade_twist[section])

        for i in range(len(blade_positions)):
            phi = phi_sections[section, i]
            theta_pitch = np.radians(pitch_sections[section, i])

            F_n = P_n_sections[section, i]
            F_t = P_t_sections[section, i]
            F = F_sections[section, i]

            phi_eff = phi - (theta_pitch + beta_twist)

            # Lokal tangential kraft (edgewise langs rotasjonsretning)
            tangential_force = F_t * np.cos(phi_eff) + F_n * np.sin(phi_eff)

            # Torque = tangential_force * radius * tip loss factor
            torque_sections[section, i] = tangential_force * r_central * F

    # Nå gjør vi kumulativ summasjon av torque baklengs
    rotor_torque_moments = np.zeros_like(torque_sections)

    for section in range(num_blade_sections_50 - 1, -1, -1):
        if section == num_blade_sections_50 - 1:  # Siste seksjon
            rotor_torque_moments[section, :] = torque_sections[section, :]
        else:
            rotor_torque_moments[section, :] = (
                torque_sections[section, :] + rotor_torque_moments[section + 1, :]
            )

    rotor_aerodynamic_edge_moments_all[wind_speed] = rotor_torque_moments


#%%
# Gravitasjonskonstant
g = 9.81  # m/s^2

# Initialiser dictionaries for gravitasjonsmomenter
gravitational_moments_all = {}
rotor_edge_gravity_all = {}

# Beregning av gravitasjonsmoment for alle vindhastigheter
for wind_speed in wind_speeds:
    # Initialiser array for gravitasjonsmoment
    gravitational_moments_sections = np.zeros((num_blade_sections_50, len(adjusted_blade_positions)))

    # Beregn gravitasjonsmoment for hver seksjon
    for section in range(num_blade_sections_50):
        # Midtradius for seksjonen (gjennomsnittlig radius)
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section

        # Massen til seksjonen
        m_i = blade_data["Section Mass"].iloc[section]

        # Loop gjennom hver rotorposisjon
        for i, blade_position in enumerate(adjusted_blade_positions):
            # Beregn gravitasjonsmoment for denne posisjonen
            gravitational_moment = m_i * g * r_central * np.cos(np.radians(blade_position))
            gravitational_moments_sections[section, i] = gravitational_moment

    # Lagre gravitasjonsmoment for denne vindhastigheten
    gravitational_moments_all[wind_speed] = gravitational_moments_sections

    # Beregn kumulative gravitasjonsmomenter (iterer baklengs)
    rotor_gravitational_edge_moments = np.zeros_like(gravitational_moments_sections)
    for section in range(num_blade_sections_50 - 1, -1, -1):  # Fra siste til første seksjon
        if section == num_blade_sections_50 - 1:  # Siste seksjon
            rotor_gravitational_edge_moments[section, :] = gravitational_moments_sections[section, :]
        else:
            rotor_gravitational_edge_moments[section, :] = (
                gravitational_moments_sections[section, :] + rotor_gravitational_edge_moments[section + 1, :]
            )

    # Lagre kumulative gravitasjonsmomenter
    rotor_edge_gravity_all[wind_speed] = rotor_gravitational_edge_moments



#%%
# Initialiser dictionary for totale kumulative edgewise-momenter
rotor_total_edgewise_moments_all = {}

# Kombiner kumulative torque- og gravitasjonsmomenter for totalt edgewise-moment
for wind_speed in wind_speeds:
    if (wind_speed in rotor_aerodynamic_edge_moments_all) and (wind_speed in rotor_edge_gravity_all):
        # Hent kumulative torque- og gravitasjonsmomenter
        rotor_aerodynamic_edge = rotor_aerodynamic_edge_moments_all[wind_speed]
        rotor_gravitational_edge = rotor_edge_gravity_all[wind_speed]
        
        # Beregn totalt kumulativt moment
        rotor_total_edgewise_moments = rotor_aerodynamic_edge + rotor_gravitational_edge
        
        # Lagre totalt kumulativt moment
        rotor_total_edgewise_moments_all[wind_speed] = rotor_total_edgewise_moments

#%%


# --- 1. Beregn gjennomsnittlig edgewise-moment fra modellen ---
average_edgewise_moments_per_wind_speed = []

for wind_speed, total_edgewise_moment in edgewise_moments_all.items():
    average_moment_per_position = np.mean(total_edgewise_moment, axis=1)  # Over rotasjon
    total_average_moment = np.sum(average_moment_per_position)  # Summer alle seksjoner
    average_edgewise_moments_per_wind_speed.append((wind_speed, total_average_moment))

average_edgewise_moments_per_wind_speed = np.array(average_edgewise_moments_per_wind_speed)
model_wind_speeds = average_edgewise_moments_per_wind_speed[:, 0]
model_moments = average_edgewise_moments_per_wind_speed[:, 1] * 1e-6  # Nm → MNm

# --- 2. Les inn baseline edgewise-moment (allerede i MNm) ---
baseline_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Baseline/baseline_edgewise.csv'
baseline_df = pd.read_csv(baseline_path)

baseline_wind_speeds = baseline_df['Wind Speed (m/s)'].values
baseline_moments = baseline_df['Average Edgewise Bending Moment (MNm)'].values  * 1e-6

# --- 3. Plot sammenligning ---
plt.figure(figsize=(10, 5))
plt.plot(model_wind_speeds, model_moments, marker='o', linestyle='-', label=f'Alpha = {alpha}')
plt.plot(baseline_wind_speeds, baseline_moments, marker='x', linestyle='--', label='Homogeneous baseline')

plt.xlabel("Wind Speed (m/s)", fontsize=16)
plt.ylabel("(MNm)", fontsize=16)
plt.title("Average Edgewise Bending Moment", fontsize=18)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
plt.xlim(3, 25)
#plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/average_edge.pdf', dpi = 300)

plt.show()

#%%
# --- 1. Filtrer bort lave vindhastigheter (< 3 m/s) ---
mask = (baseline_wind_speeds >= 3.5) & (model_wind_speeds >= 3.5)

# Interpoler baseline til samme vindhastigheter som modellen (hvis nødvendig)
baseline_interp = np.interp(model_wind_speeds, baseline_wind_speeds, baseline_moments)

# Lag maske basert på model_wind_speeds
valid_mask = model_wind_speeds >= 3.5
wind_speeds_filtered = model_wind_speeds[valid_mask]
model_moments_filtered = model_moments[valid_mask]
baseline_interp_filtered = baseline_interp[valid_mask]

# --- 2. Plot normalisert edgewise moment ---
plt.figure(figsize=(10, 5))

plt.plot(
    wind_speeds_filtered, 
    model_moments_filtered / baseline_interp_filtered, 
    marker='o', linestyle='-', label=f'Alpha = {alpha}', color='tab:blue'
)

# Legg til horisontal linje for baseline
plt.axhline(1.0, color='gray', linestyle='--', linewidth=2, label='Homogeneous baseline')

# --- 3. Stil og layout ---
plt.xlabel("Wind Speed (m/s)", fontsize=16)
plt.title("Normalized Edgewise Bending Moment Compared to Baseline", fontsize=18)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
plt.xlim(3, 25)
#plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/normalized_average_edge.pdf', dpi = 300)

plt.show()

# Lag DataFrame med filtrerte og normaliserte edgewise-momenter
normalized_edgewise_df = pd.DataFrame({
    'Wind Speed (m/s)': wind_speeds_filtered,
    'Normalized Edgewise Moment (-)': model_moments_filtered / baseline_interp_filtered
})

# Angi lagringssti
save_path_edge = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Baseline/Alfa_verdier/Edge/normalized_edgewise_alpha_{alpha}.csv'

# Lagre til CSV
normalized_edgewise_df.to_csv(save_path_edge, index=False)
print(f"Saved normalized edgewise moments to: {save_path_edge}")


#%%
# Vindhastighet for plott
wind_speed_to_plot = 15  # m/s

# Hent kumulative momenter for valgt vindhastighet
rotor_aerodynamic_edge = rotor_aerodynamic_edge_moments_all[wind_speed_to_plot]
rotor_gravitational_edge = rotor_edge_gravity_all[wind_speed_to_plot]
rotor_total_edgewise_moments = rotor_total_edgewise_moments_all[wind_speed_to_plot]

# Velg seksjon for plott (innerste seksjon, nærmest navet)
section_to_plot = 0  # Første seksjon

# Opprett plott
plt.figure(figsize=(10, 6))

# Torque moment
plt.plot(
    blade_positions, 
    rotor_aerodynamic_edge[section_to_plot, :] * 1e-6, 
    linestyle='-', label="Aerodynamic Forces", color='blue'
)

# Gravitational moment
plt.plot(
    blade_positions, 
    rotor_gravitational_edge[section_to_plot, :] * 1e-6, 
    linestyle='--', label="Gravitational", color='green'
)

# Total moment
plt.plot(
    blade_positions, 
    rotor_total_edgewise_moments[section_to_plot, :] * 1e-6, 
    linestyle='-.', label="Total", color='red'
)

# Plotinnstillinger
plt.xlabel("Blade Position (degrees)", fontsize=18)
plt.ylabel("(MNm)", fontsize=18)
plt.title(f"Edgewise Bending Moments Over One Full Rotation at {wind_speed_to_plot} m/s", fontsize=20)
plt.legend(fontsize=17, title="Bending Moment Components", title_fontsize=18)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

# Lagre plott (valgfritt)
save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Bøymoment/Edge/edgewise_moments_inner_section_full_rotation.png"
plt.savefig(save_path, dpi=300)

# Vis plott
plt.show()







#%%
#
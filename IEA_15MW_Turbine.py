import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the folder path where your files are stored
folder_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/'

# Initialize an empty list to hold the file paths
file_paths = []

# Use a loop to generate the file paths for all 50 files (from 0 to 49)
for i in range(50):
    # Construct the filename dynamically
    file_name = f'IEA-15-240-RWT_AeroDyn15_Polar_{i:02d}.dat'
    # Combine the folder path with the filename to get the full file path
    full_path = os.path.join(folder_path, file_name)
    # Add the full file path to the list
    file_paths.append(full_path)



# Initialize dictionaries to hold aerodynamic coefficients for each airfoil
aero_data = {file_path: {'aoa': [], 'cl': [], 'cd': [], 'cm': []} for file_path in file_paths}

# Function to extract aerodynamic coefficients from a file
def extract_aero_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

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
                aero_data[file_path]['aoa'].append(aoa_value)
                aero_data[file_path]['cl'].append(cl_value)
                aero_data[file_path]['cd'].append(cd_value)
                aero_data[file_path]['cm'].append(cm_value)
            except ValueError:
                # Skip lines that contain invalid float values
                continue

# Extract data for all airfoils
for path in file_paths:
    extract_aero_data(path)

# Define a mapping from index to airfoil name
airfoil_names = {
    1: 'Circular',
    8: 'SNL-FFA-W3-500',
    12: 'FFA-W3-360',
    17: 'FFA-W3-330blend',
    22: 'FFA-W3-301',
    27: 'FFA-W3-270blend',
    33: 'FFA-W3-241',
    41: 'FFA-W3-211'
}

# Select specific airfoil indices
selected_indices = [1, 8, 12, 17, 22, 27, 33, 41]
selected_airfoils = [list(aero_data.keys())[i] for i in selected_indices]

# Folder path for saving figures
save_folder = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data/'

# Plot Lift Coefficient (Cl)
plt.figure(figsize=(12, 8))
for i, path in enumerate(selected_airfoils):
    data = aero_data[path]
    aoa = np.array(data['aoa'])
    cl = np.array(data['cl'])
    plt.plot(aoa, cl, label=airfoil_names[selected_indices[i]])  # Use airfoil name for the label

plt.xlabel('Angle of Attack (degrees)', fontsize = 18)
plt.ylabel('Lift Coefficient ($C_l$)', fontsize = 18)
plt.title('Lift Coefficient vs Angle of Attack', fontsize = 20)
plt.grid()
plt.xlim(-15, 15)
# Make the tick numbers larger
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize = 16)
plt.savefig(save_folder + 'lift_coefficient_selected_plot_2.png')
plt.show()

# Plot Drag Coefficient (Cd)
plt.figure(figsize=(12, 8))
for i, path in enumerate(selected_airfoils):
    data = aero_data[path]
    aoa = np.array(data['aoa'])
    cd = np.array(data['cd'])
    plt.plot(aoa, cd, label=airfoil_names[selected_indices[i]])  # Use airfoil name for the label

plt.xlabel('Angle of Attack (degrees)', fontsize = 18)
plt.ylabel('Drag Coefficient ($C_d$)', fontsize = 18)
plt.title('Drag Coefficient vs Angle of Attack', fontsize = 20)
# Make the tick numbers larger
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.legend(fontsize = 16)
plt.xlim(-15, 90)
plt.savefig(save_folder + 'drag_coefficient_selected_plot.png')
plt.show()


# Calculate Lift-to-Drag Ratio and Plot
plt.figure(figsize=(12, 8))
for i, path in enumerate(selected_airfoils):
    data = aero_data[path]
    aoa = np.array(data['aoa'])
    cl = np.array(data['cl'])
    cd = np.array(data['cd'])
    
    # Calculate Lift-to-Drag ratio
    lift_to_drag = cl / cd
    
    plt.plot(aoa, lift_to_drag, label=airfoil_names[selected_indices[i]])  # Use airfoil name for the label

plt.xlabel('Angle of Attack (degrees)', fontsize = 18)
plt.ylabel('Lift-to-Drag Ratio ($C_l$/$C_d$)', fontsize = 18)
plt.title('Lift-to-Drag Ratio vs Angle of Attack', fontsize = 20)
plt.grid()
# Make the tick numbers larger
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize = 16)
plt.xlim(-15, 90)
plt.savefig(save_folder + 'lift_to_drag_ratio_selected_plot.png')
plt.show()

plt.figure(figsize=(12, 8))
for i, path in enumerate(selected_airfoils):
    data = aero_data[path]
    aoa = np.array(data['aoa'])  # Angle of attack
    cl = np.array(data['cl'])
    cd = np.array(data['cd'])
    
    # Filter the data for angles of attack between -30 and 30 degrees
    valid_indices = (aoa >= -30) & (aoa <= 30)
    cl_filtered = cl[valid_indices]
    cd_filtered = cd[valid_indices]
    
    # Plot Cl vs Cd for each airfoil
    plt.plot(cd_filtered, cl_filtered, label=airfoil_names[selected_indices[i]])

plt.xlabel('Drag Coefficient ($C_d$)', fontsize = 18)
plt.ylabel('Lift Coefficient ($C_l$)', fontsize = 18)
plt.title('Lift Coefficient vs Drag Coefficient', fontsize = 20)
plt.grid()
plt.xlim(0, 0.20)
# Make the tick numbers larger
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 16)  # Move the legend outside the plot
plt.tight_layout()
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data/lift_vs_drag_coefficient_filtered_plot.png')
plt.show()

#%%
# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(22, 16), sharex=False, sharey=False)

# Define larger font sizes
title_fontsize = 27
label_fontsize = 25
tick_fontsize = 22
legend_fontsize = 27

# Plot 1: Lift Coefficient (Cl)
for i, path in enumerate(selected_airfoils):
    data = aero_data[path]
    aoa = np.array(data['aoa'])
    cl = np.array(data['cl'])
    axs[0, 0].plot(aoa, cl, label=airfoil_names[selected_indices[i]])

axs[0, 0].set_xlabel('Angle of Attack (degrees)', fontsize=label_fontsize)
axs[0, 0].set_ylabel('Lift Coefficient ($C_l$)', fontsize=label_fontsize)
axs[0, 0].set_title('Lift Coefficient vs Angle of Attack', fontsize=title_fontsize)
axs[0, 0].grid()
axs[0, 0].set_xlim(-15, 90)
axs[0, 0].tick_params(axis='both', labelsize=tick_fontsize)

# Plot 2: Drag Coefficient (Cd)
for i, path in enumerate(selected_airfoils):
    data = aero_data[path]
    aoa = np.array(data['aoa'])
    cd = np.array(data['cd'])
    axs[0, 1].plot(aoa, cd, label=airfoil_names[selected_indices[i]])

axs[0, 1].set_xlabel('Angle of Attack (degrees)', fontsize=label_fontsize)
axs[0, 1].set_ylabel('Drag Coefficient ($C_d$)', fontsize=label_fontsize)
axs[0, 1].set_title('Drag Coefficient vs Angle of Attack', fontsize=title_fontsize)
axs[0, 1].grid()
axs[0, 1].set_xlim(-15, 90)
axs[0, 1].tick_params(axis='both', labelsize=tick_fontsize)

# Plot 3: Lift-to-Drag Ratio (Cl/Cd)
for i, path in enumerate(selected_airfoils):
    data = aero_data[path]
    aoa = np.array(data['aoa'])
    cl = np.array(data['cl'])
    cd = np.array(data['cd'])
    lift_to_drag = cl / cd
    axs[1, 0].plot(aoa, lift_to_drag, label=airfoil_names[selected_indices[i]])

axs[1, 0].set_xlabel('Angle of Attack (degrees)', fontsize=label_fontsize)
axs[1, 0].set_ylabel('Lift-to-Drag Ratio ($C_l$/$C_d$)', fontsize=label_fontsize)
axs[1, 0].set_title('Lift-to-Drag Ratio vs Angle of Attack', fontsize=title_fontsize)
axs[1, 0].grid()
axs[1, 0].set_xlim(-15, 90)
axs[1, 0].tick_params(axis='both', labelsize=tick_fontsize)

# Plot 4: Lift Coefficient vs Drag Coefficient (Cl vs Cd)
for i, path in enumerate(selected_airfoils):
    data = aero_data[path]
    aoa = np.array(data['aoa'])
    cl = np.array(data['cl'])
    cd = np.array(data['cd'])
    
    # Filter the data for angles of attack between -30 and 30 degrees
    valid_indices = (aoa >= -30) & (aoa <= 30)
    cl_filtered = cl[valid_indices]
    cd_filtered = cd[valid_indices]
    
    axs[1, 1].plot(cd_filtered, cl_filtered, label=airfoil_names[selected_indices[i]])

axs[1, 1].set_xlabel('Drag Coefficient ($C_d$)', fontsize=label_fontsize)
axs[1, 1].set_ylabel('Lift Coefficient ($C_l$)', fontsize=label_fontsize)
axs[1, 1].set_title('Lift Coefficient vs Drag Coefficient', fontsize=title_fontsize)
axs[1, 1].grid()
axs[1, 1].set_xlim(0, 0.20)
axs[1, 1].tick_params(axis='both', labelsize=tick_fontsize)

# Adjust layout to increase space between subplots
fig.subplots_adjust(wspace=0.6, hspace=0.8)

# Add a common legend at the bottom with larger font size
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=legend_fontsize, ncol=4)
fig.tight_layout(rect=[0, 0.1, 1.1, 0.95])  # Adjust layout to make space for the legend

# Save the combined figure with high resolution
output_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data/airfoil_subplots_spaced.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Show plot
plt.show()

#%%

# Filter and print Cl values for angles of attack between -20 and 20 degrees
for path, data in aero_data.items():
    aoa = np.array(data['aoa'])
    cl = np.array(data['cl'])
    
    # Get indices where aoa is between -20 and 20 degrees
    indices = np.where((aoa >= -20) & (aoa <= 20))
    
    # Filter the aoa and cl values based on the indices
    filtered_aoa = aoa[indices]
    filtered_cl = cl[indices]
    
    # Print the filtered values
    print(f"Angle of Attack (AoA) and Lift Coefficients (Cl) for {path.split('/')[-1]}:")
    for angle, lift in zip(filtered_aoa, filtered_cl):
        print(f" AoA: {angle:.2f}°, Cl: {lift:.4f}")
    print()  # Print a newline for better readability

#%%
import csv

# Define the file paths
file_path_blade = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/IEA-15-240-RWT_AeroDyn15_blade.dat'
file_path_prebend = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Alle_verdier/IEA-15-240-RWT_tabular/Blade Geometry-Table 1.csv'

# Initialize lists to store extracted data
blade_span = []
blade_curve_ac = []
blade_sweep_ac = []
blade_curve_angle = []
blade_twist = []
blade_chord = []
blade_prebend = []  # New list for prebend data
blade_pitch_axis = []   # List to hold pitch axis data

# Function to read the .dat file for the blade data
def read_blade_data(file_path):
    with open(file_path, 'r') as file:
        # Skip the header lines
        for _ in range(6):  # Adjust the number based on how many header lines to skip
            file.readline()
        
        # Read and extract data
        for line_number, line in enumerate(file, start=7):  # Start at line 7 (after skipping headers)
            values = line.split()
            if len(values) >= 6:  # Ensure there are enough values in the line
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


def read_prebend_data(file_path):
    # Skipping initial rows and focusing on the 'Profile' section
    profile_data = pd.read_csv(file_path, delimiter=';', skiprows=12)

    # Rename columns and clean the numeric data
    profile_data.columns = ['Spanwise position [r/R]', 'Chord [m]', 'Twist [rad]', 'Pitch axis [x/c]', 'Span [m]', 'Prebend [m]', 'Sweep [m]']
    profile_data_cleaned = profile_data.drop(0)
    
    # Replace commas with periods and convert columns to numeric
    columns_to_convert = ['Spanwise position [r/R]', 'Chord [m]', 'Twist [rad]', 'Pitch axis [x/c]', 'Span [m]', 'Prebend [m]', 'Sweep [m]']
    for col in columns_to_convert:
        profile_data_cleaned[col] = profile_data_cleaned[col].str.replace(',', '.').astype(float)
    
    # Append the prebend data to the blade_prebend list
    blade_prebend.extend(profile_data_cleaned['Prebend [m]'].tolist())
    
    # Append the pitch axis data to the blade_pitch_axis list
    blade_pitch_axis.extend(profile_data_cleaned['Pitch axis [x/c]'].tolist())

# Read the blade data
read_blade_data(file_path_blade)

# Read the prebend data
read_prebend_data(file_path_prebend)

# Print the results
print("Blade Span:", blade_span)
print("Blade Curve AC:", blade_curve_ac)
print("Blade Sweep AC:", blade_sweep_ac)
print("Blade Curve Angle:", blade_curve_angle)
print("Blade Twist:", blade_twist)
print("Blade Chord:", blade_chord)
print("Blade Prebend:", blade_prebend)  # Print the prebend values
print('Blade pitch axis', blade_pitch_axis)


print(len(blade_prebend))


#%% 
# Convert lists to numpy arrays for plotting
blade_span = np.array(blade_span)
blade_chord = np.array(blade_chord)

# Create a simple plot of chord length along the blade span
plt.figure(figsize=(13, 6))
plt.plot(blade_span, blade_chord, linestyle='-', color='b', linewidth=3)  # Adjusted linewidth for thicker line

# Set labels and title
plt.xlabel('Blade Span (m)', fontsize = 24)
plt.ylabel('Chord Length (m)', fontsize = 24)
plt.title('Chord Length along Blade Span', fontsize = 27)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.grid()
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data/chord_plot.png')

# Show plot
plt.show()

#%%
# Convert lists to numpy arrays for plotting (assuming blade_span and blade_twist are lists)
blade_span = np.array(blade_span)
blade_twist = np.array(blade_twist)

# Create a simple plot of twist angle along the blade span
plt.figure(figsize=(13, 6))
plt.plot(blade_span, blade_twist, linestyle='-', color='r', linewidth=3)

# Set labels and title
plt.xlabel('Blade Span (m)',fontsize = 24)
plt.ylabel('Twist Angle (degrees)', fontsize = 24)
plt.title('Twist Angle along Blade Span', fontsize = 27)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid()
# Optionally, save the figure
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data/twist_plot.png')

# Show plot
plt.show()


#%%

# Convert lists to numpy arrays for plotting (assuming prebend is provided)
blade_span = np.array(blade_span)
prebend = np.array(blade_prebend)  # Assuming prebend data is available

prebend = prebend[:50]
# Create a simple plot of prebend along the blade span
plt.figure(figsize=(13, 6))

# Plot prebend
plt.plot(blade_span, prebend, linestyle='-', color='g', linewidth=3, label='Prebend')

# Set labels and title
plt.xlabel('Blade Span (m)', fontsize = 24)
plt.ylabel('Prebend (m)', fontsize = 24)
plt.title('Prebend along Blade Span', fontsize = 27)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid()

# Save the plot
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data/prebend_plot.png')

# Show plot
plt.show()

#%%

# Filstien til ElastoDyn-bladdata
file_path_blade_data = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/IEA-15-240-RWT_ElastoDyn_blade.dat'

# Les inn filen og finn tabellen
try:
    with open(file_path_blade_data, 'r') as file:
        lines = file.readlines()
    
    # Finn starten og slutten av tabellen
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
    if start_index is not None and end_index is not None:
        blade_data = pd.read_csv(
            file_path_blade_data,
            skiprows=start_index,
            nrows=end_index - start_index,
            sep=r'\s+',
            header=None,
            names=["BlFract", "PitchAxis", "StrcTwst", "BMassDen", "FlpStff", "EdgStff"],
            engine='python'
        )
        print("Bladdata lastet inn:")
        print(blade_data.head())
    else:
        print("Fant ikke tabellen i filen.")
except Exception as e:
    print(f"En feil oppstod: {e}")

# Beregning av massen til hver seksjon
try:
    # Total bladlengde i meter
    blade_length = 117  # Fra spesifikasjoner

    # Beregn lengden på hver seksjon
    blade_data["Section Length"] = np.diff(
        np.append(blade_data["BlFract"].values, [1.0])
    ) * blade_length

    # Beregn massen for hver seksjon
    blade_data["Section Mass"] = blade_data["BMassDen"] * blade_data["Section Length"]

    # Summér totalmassen for kontroll
    total_mass = blade_data["Section Mass"].sum()

    # Vis resultatene
    print("\nMassen per seksjon:")
    print(blade_data[["BlFract", "Section Length", "BMassDen", "Section Mass"]])
    print(f"\nTotal bladmasse beregnet: {total_mass:.2f} kg")
except Exception as e:
    print(f"En feil oppstod under beregningene: {e}")

#%%

# Create a figure with three subplots
fig, axs = plt.subplots(4, 1, figsize=(14, 15), sharex=True)

# Plot 1: Chord Length
axs[0].plot(blade_span, blade_chord, linestyle='-', marker = 'o', color='b', linewidth=3)
axs[0].set_ylabel('m', fontsize=26)
axs[0].set_title('Chord Length', fontsize=27)
axs[0].grid()
axs[0].tick_params(axis='both', labelsize=25)

# Plot 2: Twist Angle
axs[1].plot(blade_span, blade_twist, linestyle='-',marker = 'o', color='r', linewidth=3)
axs[1].set_ylabel('Degrees', fontsize=26)
axs[1].set_title('Twist Angle', fontsize=27)
axs[1].grid()
axs[1].tick_params(axis='both', labelsize=25)

# Plot 3: Prebend
axs[2].plot(blade_span, prebend, linestyle='-', color='g',marker = 'o', linewidth=3)
axs[2].set_ylabel('m', fontsize=26)
axs[2].set_title('Prebend', fontsize=27)
axs[2].grid()
axs[2].tick_params(axis='both', labelsize=25)

# Plot 3: Prebend
axs[3].plot(blade_span,  blade_data["BMassDen"], linestyle='-', color='y',marker = 'o', linewidth=3)
axs[3].set_xlabel('Blade Span (m)', fontsize=26)
axs[3].set_ylabel('$kg/m^3$', fontsize=26)
axs[3].set_title('Section Blade Density', fontsize=27)
axs[3].grid()
axs[3].tick_params(axis='both', labelsize=25)

# Adjust layout and save the figure
fig.tight_layout()
output_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data/blade_properties_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Show plot
plt.show()

#%% Airfoil cross sections

def plot_airfoil(airfoil_number):
    # Ensure the input number is within the valid range
    if airfoil_number < 0 or airfoil_number > 49:
        print("Invalid airfoil number. Please enter a number between 0 and 49.")
        return
    
    # Updated file path
    file_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Airfoil_shape/IEA-15-240-RWT_AF{airfoil_number:02d}_Coords.txt'
    
    x_coords = []
    y_coords = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_reading = False
        
        for line in lines:
            # Skip the reference point section and start reading the actual airfoil shape
            if 'coordinates of airfoil shape' in line:
                start_reading = True
                continue  # Skip the line that contains "coordinates of airfoil shape"
            
            if start_reading:
                # Read x/c and y/c coordinates
                try:
                    x, y = map(float, line.split())
                    x_coords.append(x)
                    y_coords.append(y)
                except ValueError:
                    continue  # In case there are any lines without data

    # Plotting the airfoil shape
    plt.figure(figsize=(8, 4))
    plt.plot(x_coords, y_coords, label=f'Airfoil {airfoil_number}', color='blue')
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.title(f'Airfoil {airfoil_number} Cross Section')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example: Call the function with an airfoil number, e.g., 15
plot_airfoil(35)

#%%

def plot_airfoil_clean(airfoil_number):
    if airfoil_number < 0 or airfoil_number > 49:
        print("Invalid airfoil number. Please enter a number between 0 and 49.")
        return

    file_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Airfoil_shape/IEA-15-240-RWT_AF{airfoil_number:02d}_Coords.txt'
    
    x_coords, y_coords = [], []

    with open(file_path, 'r') as file:
        start_reading = False
        for line in file:
            if 'coordinates of airfoil shape' in line:
                start_reading = True
                continue
            if start_reading:
                try:
                    x, y = map(float, line.split())
                    x_coords.append(x)
                    y_coords.append(y)
                except ValueError:
                    continue

    plt.figure(figsize=(8, 4))
    plt.plot(x_coords, y_coords, color='black')
    plt.axis('equal')
    plt.axis('off')  # Fjerner akser, tall og rammer
    plt.tight_layout(pad=0)
    plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/airfoil.pdf', dpi = 300)
    plt.show()

# Eksempel
plot_airfoil_clean(35)


#%% compare airfoils normolized
def plot_multiple_airfoils(airfoil_numbers):
    # Dictionary to map airfoil numbers to names
    airfoil_names = {
        1: 'Circular',
        8: 'SNL-FFA-W3-500',
        12: 'FFA-W3-360',
        17: 'FFA-W3-330blend',
        22: 'FFA-W3-301',
        27: 'FFA-W3-270blend',
        33: 'FFA-W3-241',
        41: 'FFA-W3-211'
    }
    
    # Set the font size larger for the plot
    plt.rcParams.update({'font.size': 12})
    
    plt.figure(figsize=(8, 6))
    
    for airfoil_number in airfoil_numbers:
        # Ensure the input number is within the valid range
        if airfoil_number < 0 or airfoil_number > 49:
            print(f"Invalid airfoil number: {airfoil_number}. Please enter a number between 0 and 49.")
            continue
        
        # Updated file path
        file_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Airfoil_shape/IEA-15-240-RWT_AF{airfoil_number:02d}_Coords.txt'
        
        x_coords = []
        y_coords = []
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            start_reading = False
            
            for line in lines:
                # Skip the reference point section and start reading the actual airfoil shape
                if 'coordinates of airfoil shape' in line:
                    start_reading = True
                    continue  # Skip the line that contains "coordinates of airfoil shape"
                
                if start_reading:
                    # Read x/c and y/c coordinates
                    try:
                        x, y = map(float, line.split())
                        x_coords.append(x)
                        y_coords.append(y)
                    except ValueError:
                        continue  # In case there are any lines without data

        # Use the airfoil name from the dictionary
        airfoil_name = airfoil_names.get(airfoil_number, f'Airfoil {airfoil_number}')
        
        # Plot the airfoil shape
        plt.plot(x_coords, y_coords, label=airfoil_name)
    
    plt.xlabel('x/c', fontsize = 15)
    plt.ylabel('y/c', fontsize = 15)
    plt.title('Airfoil Cross Sections', fontsize = 20)
    plt.grid(True)
    plt.axis('equal')
    plt.ylim(-0.6, 0.6)
    plt.xlim(-0.1,1.1)
    
    # Move the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 13)
    plt.tight_layout()
    
    # Save the figure to the specified folder
    save_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data/airfoil_comparison.png'
    plt.savefig(save_path)
    
    plt.show()

# Example: Call the function with a list of airfoil numbers to compare
plot_multiple_airfoils([1, 8, 12, 17, 22, 27, 33, 41]) 


#%% compare airfoils
def plot_multiple_airfoils_chord(airfoil_numbers, blade_chord):
    plt.figure(figsize=(8, 4))
    
    for airfoil_number in airfoil_numbers:
        # Ensure the input number is within the valid range
        if airfoil_number < 0 or airfoil_number > 49:
            print(f"Invalid airfoil number: {airfoil_number}. Please enter a number between 0 and 49.")
            continue
        
        # Updated file path
        file_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Airfoil_shape/IEA-15-240-RWT_AF{airfoil_number:02d}_Coords.txt'
        
        x_coords = []
        y_coords = []
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            start_reading = False
            
            for line in lines:
                # Skip the reference point section and start reading the actual airfoil shape
                if 'coordinates of airfoil shape' in line:
                    start_reading = True
                    continue  # Skip the line that contains "coordinates of airfoil shape"
                
                if start_reading:
                    # Read x/c and y/c coordinates
                    try:
                        x, y = map(float, line.split())
                        x_coords.append(x)
                        y_coords.append(y)
                    except ValueError:
                        continue  # In case there are any lines without data

        # Scale the airfoil by the corresponding chord length
        chord_length = blade_chord[airfoil_number]  # Use airfoil number to index the correct chord length
        print(f'Chord length for airfoil {airfoil_number}: {chord_length}')
        x_coords = np.array(x_coords) * chord_length
        y_coords = np.array(y_coords) * chord_length

        # Plot the airfoil shape
        plt.plot(x_coords, y_coords, label=f'Airfoil {airfoil_number}')
    
    plt.xlabel('x (scaled by chord)')
    plt.ylabel('y (scaled by chord)')
    plt.title('Comparison of Non-Normalized Airfoil Cross Sections')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()

#Call the function with a list of airfoil numbers to compare
airfoil_numbers = [0,11, 46]  

# Use the blade_chord values extracted from the IEA-15-240-RWT_AeroDyn15_blade.dat file
plot_multiple_airfoils_chord(airfoil_numbers, blade_chord)



#%% airfoil blade

from mpl_toolkits.mplot3d import Axes3D


def plot_filled_blade(blade_span, blade_chord, blade_twist, blade_sweep_ac, blade_curve_angle, blade_prebend, blade_pitch_axis):
    # Increase the figure size for a larger plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize arrays for the surface coordinates
    x_surface = []
    y_surface = []
    z_surface = []

    # Assuming we have 50 blade sections, using airfoil numbers 0-49
    airfoil_numbers = list(range(50))  # Airfoil IDs from 0 to 49
    
    for i, span in enumerate(blade_span):
        airfoil_number = airfoil_numbers[i] if i < len(airfoil_numbers) else airfoil_numbers[-1]
        
        # Load airfoil shape for this section
        file_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Airfoil_shape/IEA-15-240-RWT_AF{airfoil_number:02d}_Coords.txt'
        
        x_coords = []
        y_coords = []
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            start_reading = False
            
            for line in lines:
                if 'coordinates of airfoil shape' in line:
                    start_reading = True
                    continue
                if start_reading:
                    try:
                        x, y = map(float, line.split())
                        x_coords.append(x)
                        y_coords.append(y)
                    except ValueError:
                        continue

        # Scale the airfoil by the chord length
        chord_length = blade_chord[i]
        x_coords = np.array(x_coords) * chord_length
        y_coords = np.array(y_coords) * chord_length

        # Apply pitch axis offset (shift along the chord based on normalized pitch axis position)
        pitch_axis_offset = blade_pitch_axis[i] * chord_length
        x_coords -= pitch_axis_offset  # Shift the x_coords to center at the pitch axis
        
        # Apply twist (rotation around the x-axis)
        twist_angle = np.deg2rad(blade_twist[i])
        x_rotated = x_coords * np.cos(twist_angle) - y_coords * np.sin(twist_angle)
        y_rotated = x_coords * np.sin(twist_angle) + y_coords * np.cos(twist_angle)
        
        # Apply sweep (shift along the x-axis based on Blade Sweep AC)
        sweep_offset = blade_sweep_ac[i]
        x_rotated += sweep_offset

        # Apply curve angle (rotation around the z-axis for blade curvature)
        curve_angle = np.deg2rad(blade_curve_angle[i])
        x_curved = x_rotated * np.cos(curve_angle) - y_rotated * np.sin(curve_angle)
        y_curved = x_rotated * np.sin(curve_angle) + y_rotated * np.cos(curve_angle)
        
        # Apply prebend (shift along the y-axis based on Blade Prebend)
        prebend_offset = blade_prebend[i]
        y_curved -= prebend_offset  # Apply prebend to y-axis
        
        # Store the surface data for later filling
        z_coords = np.ones_like(x_coords) * span  # z_coords represent the spanwise position
        x_surface.append(x_curved)
        y_surface.append(y_curved)
        z_surface.append(z_coords)

    # Convert surface data to numpy arrays for plotting
    x_surface = np.array(x_surface)
    y_surface = np.array(y_surface)
    z_surface = np.array(z_surface)

    # Plot a filled surface using grey color
    ax.plot_surface(x_surface, z_surface, y_surface, color='grey', edgecolor='none', alpha=0.7)

    # Remove axis labels and ticks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    ax.set_xticks([])  # Hide z ticks
    ax.set_zticks([])  # Hide z ticks
    
    

    # Set the aspect ratio
    ax.set_box_aspect([0.7, 12, 0.7])  # Aspect ratio: X, Z, Y

    # Set the viewing angle
    angle = (20, 45)  # Elevation and Azimuth
    ax.view_init(elev=angle[0], azim=angle[1])  # Set the view angle
    
    save_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data/blade_geometry_overveiw.png'
    plt.savefig(save_path, dpi = 300)

    plt.show()

# Example usage:
plot_filled_blade(blade_span, blade_chord, blade_twist, blade_sweep_ac, blade_curve_angle, blade_prebend, blade_pitch_axis)



#%% more angles

def plot_filled_blade(blade_span, blade_chord, blade_twist, blade_sweep_ac, blade_curve_angle, blade_prebend, blade_pitch_axis, angles, save_path=None):
    # Loop over each angle for multiple views
    for angle in angles:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Initialize arrays for the surface coordinates
        x_surface = []
        y_surface = []
        z_surface = []

        # Assuming we have 50 blade sections, using airfoil numbers 0-49
        airfoil_numbers = list(range(50))  # Airfoil IDs from 0 to 49

        for i, span in enumerate(blade_span):
            airfoil_number = airfoil_numbers[i] if i < len(airfoil_numbers) else airfoil_numbers[-1]

            # Load airfoil shape for this section
            file_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Airfoil_shape/IEA-15-240-RWT_AF{airfoil_number:02d}_Coords.txt'

            x_coords = []
            y_coords = []

            with open(file_path, 'r') as file:
                lines = file.readlines()
                start_reading = False

                for line in lines:
                    if 'coordinates of airfoil shape' in line:
                        start_reading = True
                        continue
                    if start_reading:
                        try:
                            x, y = map(float, line.split())
                            x_coords.append(x)
                            y_coords.append(y)
                        except ValueError:
                            continue

            # Scale the airfoil by the chord length
            chord_length = blade_chord[i]
            x_coords = np.array(x_coords) * chord_length
            y_coords = np.array(y_coords) * chord_length

            # Shift airfoil based on pitch axis (normalized to chord length)
            pitch_axis_offset = blade_pitch_axis[i] * chord_length
            x_coords -= pitch_axis_offset  # Shift the airfoil so that pitch axis is at 0

            # Apply twist (rotation around the x-axis)
            twist_angle = np.deg2rad(blade_twist[i])
            x_rotated = x_coords * np.cos(twist_angle) - y_coords * np.sin(twist_angle)
            y_rotated = x_coords * np.sin(twist_angle) + y_coords * np.cos(twist_angle)

            # Apply sweep (shift along the x-axis based on Blade Sweep AC)
            sweep_offset = blade_sweep_ac[i]
            x_rotated += sweep_offset

            # Apply curve angle (rotation around the z-axis for blade curvature)
            curve_angle = np.deg2rad(blade_curve_angle[i])
            x_curved = x_rotated * np.cos(curve_angle) - y_rotated * np.sin(curve_angle)
            y_curved = x_rotated * np.sin(curve_angle) + y_rotated * np.cos(curve_angle)

            # Apply prebend (shift along the y-axis based on Blade Prebend)
            prebend_offset = blade_prebend[i]
            y_curved -= prebend_offset  # Apply prebend to y-axis
            
            # Store the surface data for later filling
            z_coords = np.ones_like(x_coords) * span  # z_coords represent the spanwise position
            x_surface.append(x_curved)
            y_surface.append(y_curved)
            z_surface.append(z_coords)

        # Convert surface data to numpy arrays for plotting
        x_surface = np.array(x_surface)
        y_surface = np.array(y_surface)
        z_surface = np.array(z_surface)

        # Plot a filled surface using grey color
        ax.plot_surface(x_surface, z_surface, y_surface, color='grey', edgecolor='none', alpha=0.7)

        # Hide the axes, ticks, and labels
        ax.set_xticks([])  # Hide x ticks

        ax.set_zticks([])  # Hide z ticks
        ax.set_xlabel('')   # Hide x label
        ax.set_ylabel('')   # Hide y label
        ax.set_zlabel('')   # Hide z label

        # Remove grid lines
        ax.grid(False)

        # Keep the aspect ratio
        ax.set_box_aspect([0.7, 12, 0.7])  # Aspect ratio: X, Z, Y

        # Set the viewing angle
        ax.view_init(elev=angle[0], azim=angle[1])

        # Save the figure if a save path is provided
        if save_path:
            filename = f'blade_geometry_elev_{angle[0]}_azim_{angle[1]}.png'
            plt.savefig(f'{save_path}/{filename}', dpi=300)
            print(f"Figure saved to {save_path}/{filename}")
        
        # Show the plot
        plt.show()

# Example usage:
save_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data'
angles = [(90, 0),(0, 0)]
plot_filled_blade(blade_span, blade_chord, blade_twist, blade_sweep_ac, blade_curve_angle, blade_prebend, blade_pitch_axis, angles, save_path)


#%%

def plot_filled_blade_with_arrows(blade_span, blade_chord, blade_twist, blade_sweep_ac, blade_curve_angle, blade_prebend, blade_pitch_axis, angles, reference_point, save_path=None):
    # Loop over each angle for multiple views
    for angle in angles:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Initialize arrays for the surface coordinates
        x_surface = []
        y_surface = []
        z_surface = []
        face_colors = []  # Colors for each section

        # Assuming we have 50 blade sections, using airfoil numbers 0-49
        airfoil_numbers = list(range(50))  # Airfoil IDs from 0 to 49

        for i, span in enumerate(blade_span):
            airfoil_number = airfoil_numbers[i] if i < len(airfoil_numbers) else airfoil_numbers[-1]

            # Load airfoil shape for this section
            file_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Airfoil_shape/IEA-15-240-RWT_AF{airfoil_number:02d}_Coords.txt'

            x_coords = []
            y_coords = []

            with open(file_path, 'r') as file:
                lines = file.readlines()
                start_reading = False

                for line in lines:
                    if 'coordinates of airfoil shape' in line:
                        start_reading = True
                        continue
                    if start_reading:
                        try:
                            x, y = map(float, line.split())
                            x_coords.append(x)
                            y_coords.append(y)
                        except ValueError:
                            continue

            # Scale the airfoil by the chord length
            chord_length = blade_chord[i]
            x_coords = np.array(x_coords) * chord_length
            y_coords = np.array(y_coords) * chord_length

            # Shift airfoil based on pitch axis (normalized to chord length)
            pitch_axis_offset = blade_pitch_axis[i] * chord_length
            x_coords -= pitch_axis_offset  # Shift the airfoil so that pitch axis is at 0

            # Apply twist (rotation around the x-axis)
            twist_angle = np.deg2rad(blade_twist[i])
            x_rotated = x_coords * np.cos(twist_angle) - y_coords * np.sin(twist_angle)
            y_rotated = x_coords * np.sin(twist_angle) + y_coords * np.cos(twist_angle)

            # Apply sweep (shift along the x-axis based on Blade Sweep AC)
            sweep_offset = blade_sweep_ac[i]
            x_rotated += sweep_offset

            # Apply curve angle (rotation around the z-axis for blade curvature)
            curve_angle = np.deg2rad(blade_curve_angle[i])
            x_curved = x_rotated * np.cos(curve_angle) - y_rotated * np.sin(curve_angle)
            y_curved = x_rotated * np.sin(curve_angle) + y_rotated * np.cos(curve_angle)

            # Apply prebend (shift along the y-axis based on Blade Prebend)
            prebend_offset = blade_prebend[i]
            y_curved -= prebend_offset  # Apply prebend to y-axis
            
            # Store the surface data for later filling
            z_coords = np.ones_like(x_coords) * span  # z_coords represent the spanwise position
            x_surface.append(x_curved)
            y_surface.append(y_curved)
            z_surface.append(z_coords)

            # Determine color based on the reference point
            if span >= reference_point:
                face_colors.append('grey')  # Mark sections from reference point outward in red
            else:
                face_colors.append('grey')  # Other sections remain grey

        # Convert surface data to numpy arrays for plotting
        x_surface = np.array(x_surface)
        y_surface = np.array(y_surface)
        z_surface = np.array(z_surface)

        # Plot each section individually with corresponding color
        for i in range(len(x_surface)):
            ax.plot_surface(
                x_surface[i:i+2], z_surface[i:i+2], y_surface[i:i+2],
                color=face_colors[i], edgecolor='none', alpha=0.7
            )

        # Add arrows for edgewise and flapwise moments
        arrow_span = blade_span[-1] * 0.9  # Position arrows at 90% of blade length
        ax.quiver(0, arrow_span, -3, 0, -1, -17, color='blue', label='Flapwise', arrow_length_ratio=0.1)
        ax.quiver(0, arrow_span, -3, 0, -2, 14, color='blue', label='Flapwise', arrow_length_ratio=0.1)
        ax.quiver(0, arrow_span, -3, 20, -2, 0, color='red', label='Edgewise', arrow_length_ratio=0.1)
        ax.quiver(0, arrow_span, -3, -20, -2, 0, color='red', label='Edgewise', arrow_length_ratio=0.1)
        # Hide the axes, ticks, and labels
        ax.axis('off')  # Turn off the entire axis
        
        

        # Keep the aspect ratio
        ax.set_box_aspect([0.7, 12, 0.7])  # Aspect ratio: X, Z, Y

        # Set the viewing angle
        ax.view_init(elev=angle[0], azim=angle[1])
        
        #ax.set_title('Edgewise Bending Moment', fontsize = 20, pad = -20)

        # Save the figure if a save path is provided
        if save_path:
            filename = f'blade_geometry_elev_{angle[0]}_azim_{angle[1]}_with_arrows.png'
            plt.savefig(f'{save_path}/{filename}', dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"Figure saved to {save_path}/{filename}")
        
        # Show the plot
        plt.show()

# Example usage:
save_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data'
angles = [(20, 45)]
reference_point = 0.6 * 120  # Assuming rotor radius is 120 meters
plot_filled_blade_with_arrows(
    blade_span, blade_chord, blade_twist, blade_sweep_ac, 
    blade_curve_angle, blade_prebend, blade_pitch_axis, 
    angles, reference_point, save_path
)
#%% Pitch and rotor speed

# Define the file path
file_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Alle_verdier/IEA-15-240-RWT_tabular/Rotor Performance-Table 1.csv'

# Load the data from the CSV file
data_2 = pd.read_csv(file_path, sep=';', decimal=',')

def plot_pitch_vs_rotor_speed(data):
    # Extract the relevant columns
    wind_speeds = data['Wind [m/s]']
    pitch_angles = data['Pitch [deg]']
    rotor_speeds = data['Rotor Speed [rpm]']
    torque= data['Torque [MNm]']

    # Create a figure and axis with specified size
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Pitch Angles
    ax1.set_xlabel('Wind Speed [m/s]')
    ax1.set_ylabel('Pitch Angle [deg]', color='tab:blue')
    ax1.plot(wind_speeds, pitch_angles, color='tab:blue', label='Pitch Angle')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for Rotor Speed
    ax2 = ax1.twinx()
    ax2.set_ylabel('Gen. Torque [MNm]', color='tab:green')
    ax2.plot(wind_speeds, torque, color='tab:green', label='Gen. Torque')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Add vertical lines to indicate regions
    ax1.axvline(6.98, color='gray', linestyle='--')
    ax1.axvline(10.59, color='gray', linestyle='--')

    # Annotate regions
    ax1.text(3.7, 25, 'Region 1.5', rotation=0, verticalalignment='center')
    ax1.text(7.9, 25, 'Region 2', rotation=0, verticalalignment='center')
    ax1.text(16.7, 25, 'Region 3', rotation=0, verticalalignment='center')

    plt.xlim(3,25)

    # Show the legend
    fig.legend(loc='upper left', bbox_to_anchor=(0.7, 0.3))
    
    save_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data/blade_pitch_rotor_speed_vs_wind_speed.png'
    plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()

# Call the function with the loaded data
plot_pitch_vs_rotor_speed(data_2)

#%%
# Initialize the figure and the first y-axis for torque
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot calculated torque curve on the primary y-axis
ax1.plot(data_2['Wind [m/s]'], data_2['Torque [MNm]'], linestyle='-', color='blue', label="Calculated Torque")
ax1.set_xlabel("Wind Speed (m/s)")
ax1.set_ylabel("Torque (MNm)", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)
ax1.set_xlim(3, 25)  # Set x-axis limits from 3 to 25 m/s

# Create a second y-axis for blade pitch angle
ax2 = ax1.twinx()
ax2.plot(data_2['Wind [m/s]'], data_2['Pitch [deg]'], linestyle='-', color='red', label="Blade Pitch Angle")
ax2.set_ylabel("Blade Pitch Angle (degrees)", color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_xlim(3, 25)  # Apply the same x-axis limits to the twin axis

    # Add vertical lines to indicate regions
ax1.axvline(6.98, color='gray', linestyle='--')
ax1.axvline(10.59, color='gray', linestyle='--')

    # Annotate regions
ax1.text(3.7, 22, 'Region 1.5', rotation=0, verticalalignment='center')
ax1.text(7.9, 22, 'Region 2', rotation=0, verticalalignment='center')
ax1.text(16.7, 22, 'Region 3', rotation=0, verticalalignment='center')

# Title and legends
fig.suptitle("Control Regulation")

save_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/4.11.figures/control_regulation.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')


# Show the plot
plt.show()

#%%

def plot_power_and_thrust_coefficients_same_axis(data):
    # Extract the relevant columns
    wind_speeds = data['Wind [m/s]']
    power_coefficients = data['Aero Power Coefficient [-]']
    thrust_coefficients = data['Thrust Coefficient [-]']

    # Create a figure and axis with specified size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Power Coefficient
    ax.set_xlabel('Wind Speed [m/s]')

    ax.plot(wind_speeds, power_coefficients, color='tab:red', linestyle='-', label='Power Coefficient $C_P$')
    ax.plot(wind_speeds, thrust_coefficients, color='tab:blue', linestyle='--', label='Thrust Coefficient $C_T$')


    # Add vertical lines to indicate regions
    ax.axvline(6.98, color='gray', linestyle='--')
    ax.axvline(10.59, color='gray', linestyle='--')

    # Annotate regions
    ax.text(3.7, 1.05, 'Region 1.5', rotation=0, verticalalignment='center')
    ax.text(7.9, 1.05, 'Region 2', rotation=0, verticalalignment='center')
    ax.text(16.7, 1.05, 'Region 3', rotation=0, verticalalignment='center')
    
    plt.grid()
    
    plt.ylim(0, 1)
    
    plt.xlim(3,25)

    # Show the legend
    ax.legend(loc='upper right')

    # Adjust layout to ensure a clean plot
    plt.tight_layout()

    # Save the plot to the specified directory
    save_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Airfoil_data/power_and_thrust_coefficients_vs_wind_speed.png'
    plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()

# Call the function with the loaded data
plot_power_and_thrust_coefficients_same_axis(data_2)

#%% Power curve


# Function to plot and save the power curve
def plot_power_curve(data, save_path):
    # Add 0 power for wind speeds below 3 m/s and above 25 m/s
    extended_wind_speeds = [0, 3] + data['Wind [m/s]'].tolist() + [25, 26]
    extended_power_output = [0, 0] + data['Power [MW]'].tolist() + [0, 0]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(extended_wind_speeds, extended_power_output, color='tab:red', label='Power Output (MW)')
    
    # Add vertical lines for cut-in, rated, and cut-out speeds
    ax.axvline(x=3, color='blue', linestyle='--', label='Cut-in (3 m/s)')
    ax.axvline(x=10.59, color='green', linestyle='--', label='Rated (10.59 m/s)')
    ax.axvline(x=25, color='black', linestyle='--', label='Cut-out (25 m/s)')
    
    # Add labels for each region
    ax.text(0.23, 16.5, 'Region 1', rotation=0, verticalalignment='center', fontsize = 14)
    ax.text(6, 16.5, 'Region 2', rotation=0, verticalalignment='center',fontsize = 14)
    ax.text(16.7, 16.5, 'Region 3', rotation=0, verticalalignment='center',fontsize = 14)
    
    # Add labels and title
    ax.set_xlabel('Wind Speed [m/s]', fontsize = 15 )
    ax.set_ylabel('Power Output [MW]', fontsize = 15)
   # ax.set_title('Power Curve')
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=14)
   # ax.legend()

    ax.set_xlim(0, 26)  # Adjust x-axis limits to cover the extended range
    
    # Save the plot to the specified path
    plt.savefig(save_path,dpi = 300)
    
    # Show the plot
    plt.show()

# Path to save the plot
save_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/4.11.figures/power_curve_refions.png'

# Call the function to plot and save the power curve
plot_power_curve(data_2, save_path)

#%%

def plot_filled_blade_with_arrows(
    blade_span, blade_chord, blade_twist, blade_sweep_ac, 
    blade_curve_angle, blade_prebend, blade_pitch_axis, 
    angles
):
    # Loop over each angle for multiple views
    for angle in angles:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Initialize arrays for the surface coordinates
        x_surface = []
        y_surface = []
        z_surface = []

        # Assuming we have 50 blade sections, using airfoil numbers 0-49
        airfoil_numbers = list(range(50))  # Airfoil IDs from 0 to 49

        for i, span in enumerate(blade_span):
            airfoil_number = airfoil_numbers[i] if i < len(airfoil_numbers) else airfoil_numbers[-1]

            # Load airfoil shape for this section
            file_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Airfoil_shape/IEA-15-240-RWT_AF{airfoil_number:02d}_Coords.txt'

            x_coords = []
            y_coords = []

            with open(file_path, 'r') as file:
                lines = file.readlines()
                start_reading = False

                for line in lines:
                    if 'coordinates of airfoil shape' in line:
                        start_reading = True
                        continue
                    if start_reading:
                        try:
                            x, y = map(float, line.split())
                            x_coords.append(x)
                            y_coords.append(y)
                        except ValueError:
                            continue

            # Scale the airfoil by the chord length
            chord_length = blade_chord[i]
            x_coords = np.array(x_coords) * chord_length
            y_coords = np.array(y_coords) * chord_length

            # Shift airfoil based on pitch axis (normalized to chord length)
            pitch_axis_offset = blade_pitch_axis[i] * chord_length
            x_coords -= pitch_axis_offset  # Shift the airfoil so that pitch axis is at 0

            # Apply twist (rotation around the x-axis)
            twist_angle = np.deg2rad(blade_twist[i])
            x_rotated = x_coords * np.cos(twist_angle) - y_coords * np.sin(twist_angle)
            y_rotated = x_coords * np.sin(twist_angle) + y_coords * np.cos(twist_angle)

            # Apply sweep (shift along the x-axis based on Blade Sweep AC)
            sweep_offset = blade_sweep_ac[i]
            x_rotated += sweep_offset

            # Apply curve angle (rotation around the z-axis for blade curvature)
            curve_angle = np.deg2rad(blade_curve_angle[i])
            x_curved = x_rotated * np.cos(curve_angle) - y_rotated * np.sin(curve_angle)
            y_curved = x_rotated * np.sin(curve_angle) + y_rotated * np.cos(curve_angle)

            # Apply prebend (shift along the y-axis based on Blade Prebend)
            prebend_offset = blade_prebend[i]
            y_curved -= prebend_offset  # Apply prebend to y-axis
            
            # Store the surface data for later filling
            z_coords = np.ones_like(x_coords) * span  # z_coords represent the spanwise position
            x_surface.append(x_curved)
            y_surface.append(y_curved)
            z_surface.append(z_coords)

        # Convert surface data to numpy arrays for plotting
        x_surface = np.array(x_surface)
        y_surface = np.array(y_surface)
        z_surface = np.array(z_surface)

        # Plot each section individually with the same color
        for i in range(len(x_surface)):
            ax.plot_surface(
                y_surface[i:i+2], z_surface[i:i+2], x_surface[i:i+2],
                color='grey', edgecolor='none', alpha=0.7
            )

        # Hide the axes, ticks, and labels
        ax.axis('off')  # Turn off the entire axis

        # Keep the aspect ratio
        ax.set_box_aspect([0.7, 12, 0.7])  # Aspect ratio: X, Z, Y

        # Set the viewing angle to make the blade stand vertically
        ax.view_init(elev=90, azim=270)  # Look from the side to make it vertical

        # Show the plot
        plt.show()

# Example usage:
angles = [(0, 0)]  # Adjust angles as needed
plot_filled_blade_with_arrows(
    blade_span, blade_chord, blade_twist, blade_sweep_ac, 
    blade_curve_angle, blade_prebend, blade_pitch_axis, 
    angles
)


#%%
def plot_two_blades_with_individual_views(
    blade_span, blade_chord, blade_twist, blade_sweep_ac, 
    blade_curve_angle, blade_prebend, blade_pitch_axis
):
    fig = plt.figure(figsize=(16, 8))

    # Function to generate surface coordinates for a single blade
    def generate_blade_surface(blade_span, blade_chord, blade_twist, blade_sweep_ac, blade_curve_angle, blade_prebend, blade_pitch_axis, flip=False):
        x_surface = []
        y_surface = []
        z_surface = []

        # Assuming we have 50 blade sections, using airfoil numbers 0-49
        airfoil_numbers = list(range(50))  # Airfoil IDs from 0 to 49

        for i, span in enumerate(blade_span):
            airfoil_number = airfoil_numbers[i] if i < len(airfoil_numbers) else airfoil_numbers[-1]

            # Load airfoil shape for this section
            file_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Airfoil_shape/IEA-15-240-RWT_AF{airfoil_number:02d}_Coords.txt'

            x_coords = []
            y_coords = []

            with open(file_path, 'r') as file:
                lines = file.readlines()
                start_reading = False

                for line in lines:
                    if 'coordinates of airfoil shape' in line:
                        start_reading = True
                        continue
                    if start_reading:
                        try:
                            x, y = map(float, line.split())
                            x_coords.append(x)
                            y_coords.append(y)
                        except ValueError:
                            continue

            # Scale the airfoil by the chord length
            chord_length = blade_chord[i]
            x_coords = np.array(x_coords) * chord_length
            y_coords = np.array(y_coords) * chord_length

            # Shift airfoil based on pitch axis (normalized to chord length)
            pitch_axis_offset = blade_pitch_axis[i] * chord_length
            x_coords -= pitch_axis_offset  # Shift the airfoil so that pitch axis is at 0

            # Apply twist (rotation around the x-axis)
            twist_angle = np.deg2rad(blade_twist[i])
            x_rotated = x_coords * np.cos(twist_angle) - y_coords * np.sin(twist_angle)
            y_rotated = x_coords * np.sin(twist_angle) + y_coords * np.cos(twist_angle)

            # Apply sweep (shift along the x-axis based on Blade Sweep AC)
            sweep_offset = blade_sweep_ac[i]
            x_rotated += sweep_offset

            # Apply curve angle (rotation around the z-axis for blade curvature)
            curve_angle = np.deg2rad(blade_curve_angle[i])
            x_curved = x_rotated * np.cos(curve_angle) - y_rotated * np.sin(curve_angle)
            y_curved = x_rotated * np.sin(curve_angle) + y_rotated * np.cos(curve_angle)

            # Apply prebend (shift along the y-axis based on Blade Prebend)
            prebend_offset = blade_prebend[i]
            y_curved -= prebend_offset  # Apply prebend to y-axis
            
            # Store the surface data for later filling
            z_coords = np.ones_like(x_coords) * span  # z_coords represent the spanwise position

            if flip:  # Flip the blade to make it point downwards
                z_coords = -z_coords  # Mirror the z-axis

            x_surface.append(x_curved)
            y_surface.append(y_curved)
            z_surface.append(z_coords)

        return np.array(x_surface), np.array(y_surface), np.array(z_surface)

    # Generate surfaces for the two blades
    x_surface_up, y_surface_up, z_surface_up = generate_blade_surface(
        blade_span, blade_chord, blade_twist, blade_sweep_ac, blade_curve_angle, blade_prebend, blade_pitch_axis, flip=False
    )
    x_surface_down, y_surface_down, z_surface_down = generate_blade_surface(
        blade_span, blade_chord, blade_twist, blade_sweep_ac, blade_curve_angle, blade_prebend, blade_pitch_axis, flip=True
    )

    # Plot the upward blade in its own subplot
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(len(x_surface_up)):
        ax1.plot_surface(
            y_surface_up[i:i+2], z_surface_up[i:i+2], x_surface_up[i:i+2],
            color='grey', edgecolor='none', alpha=0.7
        )
    ax1.axis('off')
    ax1.set_box_aspect([0.7, 12, 0.7])
    ax1.view_init(elev=90, azim=272)  # Customize viewing angle for upward blade

    # Plot the downward blade in its own subplot
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(len(x_surface_down)):
        ax2.plot_surface(
            y_surface_down[i:i+2], z_surface_down[i:i+2], x_surface_down[i:i+2],
            color='grey', edgecolor='none', alpha=0.7
        )
    ax2.axis('off')
    ax2.set_box_aspect([0.7, 12, 0.7])
    ax2.view_init(elev=90, azim=280)  # Customize viewing angle for downward blade

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
plot_two_blades_with_individual_views(
    blade_span, blade_chord, blade_twist, blade_sweep_ac, 
    blade_curve_angle, blade_prebend, blade_pitch_axis
)


#%%


def plot_two_blades_with_custom_rotated_axes(
    blade_span, blade_chord, blade_twist, blade_sweep_ac, 
    blade_curve_angle, blade_prebend, blade_pitch_axis, save_path=None
):
    fig = plt.figure(figsize=(16, 8))

    # Function to generate surface coordinates for a single blade
    def generate_blade_surface(blade_span, blade_chord, blade_twist, blade_sweep_ac, blade_curve_angle, blade_prebend, blade_pitch_axis, flip=False):
        x_surface = []
        y_surface = []
        z_surface = []

        # Assuming we have 50 blade sections, using airfoil numbers 0-49
        airfoil_numbers = list(range(50))  # Airfoil IDs from 0 to 49

        for i, span in enumerate(blade_span):
            airfoil_number = airfoil_numbers[i] if i < len(airfoil_numbers) else airfoil_numbers[-1]

            # Load airfoil shape for this section
            file_path = f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Airfoil_shape/IEA-15-240-RWT_AF{airfoil_number:02d}_Coords.txt'

            x_coords = []
            y_coords = []

            with open(file_path, 'r') as file:
                lines = file.readlines()
                start_reading = False

                for line in lines:
                    if 'coordinates of airfoil shape' in line:
                        start_reading = True
                        continue
                    if start_reading:
                        try:
                            x, y = map(float, line.split())
                            x_coords.append(x)
                            y_coords.append(y)
                        except ValueError:
                            continue

            # Scale the airfoil by the chord length
            chord_length = blade_chord[i]
            x_coords = np.array(x_coords) * chord_length
            y_coords = np.array(y_coords) * chord_length

            # Shift airfoil based on pitch axis (normalized to chord length)
            pitch_axis_offset = blade_pitch_axis[i] * chord_length
            x_coords -= pitch_axis_offset  # Shift the airfoil so that pitch axis is at 0

            # Apply twist (rotation around the x-axis)
            twist_angle = np.deg2rad(blade_twist[i])
            x_rotated = x_coords * np.cos(twist_angle) - y_coords * np.sin(twist_angle)
            y_rotated = x_coords * np.sin(twist_angle) + y_coords * np.cos(twist_angle)

            # Apply sweep (shift along the x-axis based on Blade Sweep AC)
            sweep_offset = blade_sweep_ac[i]
            x_rotated += sweep_offset

            # Apply curve angle (rotation around the z-axis for blade curvature)
            curve_angle = np.deg2rad(blade_curve_angle[i])
            x_curved = x_rotated * np.cos(curve_angle) - y_rotated * np.sin(curve_angle)
            y_curved = x_rotated * np.sin(curve_angle) + y_rotated * np.cos(curve_angle)

            # Apply prebend (shift along the y-axis based on Blade Prebend)
            prebend_offset = blade_prebend[i]
            y_curved -= prebend_offset  # Apply prebend to y-axis
            
            # Store the surface data for later filling
            z_coords = np.ones_like(x_coords) * span  # z_coords represent the spanwise position

            if flip:  # Flip the blade to make it point downwards
                z_coords = -z_coords  # Mirror the z-axis

            x_surface.append(x_curved)
            y_surface.append(y_curved)
            z_surface.append(z_coords)

        return np.array(x_surface), np.array(y_surface), np.array(z_surface)

    # Generate surfaces for the two blades
    x_surface_up, y_surface_up, z_surface_up = generate_blade_surface(
        blade_span, blade_chord, blade_twist, blade_sweep_ac, blade_curve_angle, blade_prebend, blade_pitch_axis, flip=False
    )
    x_surface_down, y_surface_down, z_surface_down = generate_blade_surface(
        blade_span, blade_chord, blade_twist, blade_sweep_ac, blade_curve_angle, blade_prebend, blade_pitch_axis, flip=True
    )

    # Define rotation angles in radians
    rotation_angle_ax1 = np.radians(2)  # 2 degrees for the first plot
    rotation_angle_ax2 = np.radians(10)  # 10 degrees for the second plot

    # Compute rotated axes for the first plot
    x_x_rot_ax1 = 200 * np.cos(rotation_angle_ax1) - 0 * np.sin(rotation_angle_ax1)
    x_y_rot_ax1 = 200 * np.sin(rotation_angle_ax1) + 0 * np.cos(rotation_angle_ax1)
    y_x_rot_ax1 = 0 * np.cos(rotation_angle_ax1) - 200 * np.sin(rotation_angle_ax1)
    y_y_rot_ax1 = 0 * np.sin(rotation_angle_ax1) + 200 * np.cos(rotation_angle_ax1)

    # Compute rotated axes for the second plot
    x_x_rot_ax2 = 200 * np.cos(rotation_angle_ax2) - 0 * np.sin(rotation_angle_ax2)
    x_y_rot_ax2 = 200 * np.sin(rotation_angle_ax2) + 0 * np.cos(rotation_angle_ax2)
    y_x_rot_ax2 = 0 * np.cos(rotation_angle_ax2) - 200 * np.sin(rotation_angle_ax2)
    y_y_rot_ax2 = 0 * np.sin(rotation_angle_ax2) + 200 * np.cos(rotation_angle_ax2)

    # Plot the upward blade
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(len(x_surface_up)):
        ax1.plot_surface(
            y_surface_up[i:i+2], z_surface_up[i:i+2], x_surface_up[i:i+2],
            color='grey', edgecolor='none', alpha=0.7
        )
    # Add rotated axis arrows for the first plot
    ax1.quiver(0, 0, 0, x_x_rot_ax1, x_y_rot_ax1, 0, color='black', arrow_length_ratio=0.1, linewidth=2, linestyle ='--')  # Rotated x-axis
    ax1.quiver(0, 0, 0, y_x_rot_ax1, y_y_rot_ax1, 0, color='black', arrow_length_ratio=0.1, linewidth=2,linestyle ='--')  # Rotated y-axis
    ax1.axis('off')
    ax1.set_title("Top Position", fontsize=30, pad=20)
    ax1.set_box_aspect([0.7, 12, 0.7])
    ax1.view_init(elev=90, azim=272)

    # Plot the downward blade
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(len(x_surface_down)):
        ax2.plot_surface(
            y_surface_down[i:i+2], z_surface_down[i:i+2], x_surface_down[i:i+2],
            color='grey', edgecolor='none', alpha=0.7
        )
    # Add rotated axis arrows for the second plot
    ax2.quiver(0, 0, 0, x_x_rot_ax2, x_y_rot_ax2, 0, color='black', arrow_length_ratio=0.1, linewidth=2,linestyle ='--')  # Rotated x-axis
    ax2.quiver(0, 0, 0, -y_x_rot_ax2, -y_y_rot_ax2, 0, color='black', arrow_length_ratio=0.1, linewidth=2,linestyle ='--')  # Rotated y-axis (flipped)
    ax2.set_title("Bottom Position", fontsize=30, pad=20)
    ax2.axis('off')
    ax2.set_box_aspect([0.7, 12, 0.7])
    ax2.view_init(elev=90, azim=280)

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(f"{save_path}/two_blades_with_rotated_axes.png", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
plot_two_blades_with_custom_rotated_axes(
    blade_span, blade_chord, blade_twist, blade_sweep_ac, 
    blade_curve_angle, blade_prebend, blade_pitch_axis,
    save_path="/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Bøymoment"
)

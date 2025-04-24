import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from tkinter import Tk
from tkinter.filedialog import askdirectory
import os
import glob
import re

# Function to prompt the user to select a folder
def select_folder():
    """Prompt the user to select a folder containing the row profile files."""
    Tk().withdraw()  # Hide the root window
    folder_path = askdirectory(title="Select Folder with Row Profile Files")
    if not folder_path:
        raise ValueError("No folder selected. Exiting.")
    return folder_path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from tkinter import Tk
from tkinter.filedialog import askdirectory
import os
import glob
import re

# Function to prompt the user to select a folder
def select_folder():
    """Prompt the user to select a folder containing the row profile files."""
    Tk().withdraw()  # Hide the root window
    folder_path = askdirectory(title="Select Folder with Row Profile Files")
    if not folder_path:
        raise ValueError("No folder selected. Exiting.")
    return folder_path

# Function to plot the profile and smoothed curve
def plot_profile(x_data, y_data, y_smoothed, peaks, lowest_valley_x, lowest_valley_index, left_peak_x, right_peak_x, filename, show_plots):
    """Plot the profile and smoothed curve with peaks and valleys."""
    plt.figure()
    plt.plot(x_data, y_data, 'o', label='Original Data')
    plt.plot(x_data, y_smoothed, '-', label='Smoothed Data')
    plt.scatter(x_data[peaks], y_smoothed[peaks], color='red', label='Peaks', zorder=5)
    plt.scatter([lowest_valley_x], [y_smoothed[lowest_valley_index]], color='blue', label='Lowest Valley', zorder=5)
    if right_peak_x is not None:
        plt.axvline(right_peak_x, color='green', linestyle='--', label='Right Peak')
    if left_peak_x is not None:
        plt.axvline(left_peak_x, color='purple', linestyle='--', label='Left Peak')
    plt.title(f'Profile and Smoothed Data for {os.path.basename(filename)}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    if show_plots:
        plt.show()
    else:
        plt.close()

def process_file(filename, show_plots=None):
    """Process and analyze a single row profile file."""
    # Load the data from the file
    y_data = np.loadtxt(filename, dtype=int)

    # Ensure the data is a 1-D array
    if y_data.ndim > 1:
        y_data = y_data.flatten()

    # Generate x_data based on the length of y_data
    x_data = np.linspace(0, 5, num=len(y_data))

    # Apply Gaussian filter
    sigma = 2  # Standard deviation of the Gaussian kernel
    y_smoothed = gaussian_filter1d(y_data, sigma)

    # Find peaks in the smoothed data
    peaks, _ = find_peaks(y_smoothed, prominence=0.5)
    peak_heights = y_smoothed[peaks]

    # Sort peaks by height and select the two highest
    sorted_indices = np.argsort(peak_heights)[::-1]
    top_two_indices = sorted_indices[:2]
    highest_peaks = peaks[top_two_indices]
    highest_peaks_x = x_data[highest_peaks]

    # Find valleys (local minima) by inverting the smoothed data
    valleys, _ = find_peaks(-y_smoothed, prominence=0.5)
    lowest_valley_index = valleys[np.argmin(y_smoothed[valleys])]
    lowest_valley_x = x_data[lowest_valley_index]

    # Ensure the "right" peak is after the lowest valley
    right_peak_x = next((x for x in highest_peaks_x if x > lowest_valley_x), None)

    # Ensure the "left" peak is before the lowest valley
    left_peak_x = next((x for x in sorted(highest_peaks_x, reverse=True) if x < lowest_valley_x), None)

    # Plot the profile and smoothed curve if show_plots is True
    plot_profile(x_data, y_data, y_smoothed, peaks, lowest_valley_x, lowest_valley_index, left_peak_x, right_peak_x, filename, show_plots)

    # Calculate the left and right positioning
    if left_peak_x is not None and right_peak_x is not None:
        left_positioning = 3000 * (lowest_valley_x - left_peak_x) / (right_peak_x - left_peak_x)
        right_positioning = 3000 * (right_peak_x - lowest_valley_x) / (right_peak_x - left_peak_x)

        # Calculate the difference between left and right positioning
        positioning_difference = abs(left_positioning - right_positioning) / 2
        return positioning_difference
    else:
        return None

# Main script for processing multiple files
if __name__ == "__main__":
    try:
        # Prompt the user to select a folder
        folder_path = select_folder()

        # Find all row profile files in the selected folder
        file_pattern = os.path.join(folder_path, '*Profile.txt')
        files = glob.glob(file_pattern)

        if not files:
            print("No row profile files found in the selected folder.")
            exit()

        # Group files by the first digit of the three-digit number in their names
        grouped_files = {1: [], 2: [], 3: [], 4: []}
        for file in files:
            # Extract the three-digit number from the filename
            match = re.search(r'\b(\d{3})\b', os.path.basename(file))
            if match:
                first_digit = int(match.group(1)[0])  # Get the first digit of the three-digit number
                if first_digit in grouped_files:
                    grouped_files[first_digit].append(file)

        # Process each group
        for group, group_files in grouped_files.items():
            if not group_files:
                print(f"No files found for group starting with {group}.")
                continue
            positioning_differences = []

            for file in group_files:
                difference = process_file(file)  # Default behavior: show plots only if no peaks are found
                if difference is not None:
                    positioning_differences.append(difference)
                else:
                    print(f"Could not calculate positioning difference for {file} (missing peaks).")

            # Calculate average positioning difference for the group
            if positioning_differences:
                average_difference = np.mean(positioning_differences)
                print(f"\nAverage Positioning Difference for Group {group}: {average_difference:.3f}um")
            else:
                print(f"\nNo valid positioning differences calculated for Group {group}.")

    except Exception as e:
        print(f"An error occurred: {e}")
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory

from core_utils import (
    find_valley_and_fit_parabola,
    find_horizontal_segment,
    find_greatest_gradient_segments,
    find_intersection_with_horizontal,
    outer_distance,
    inner_distance
)

PIXEL_SCALE = 12.5  # Conversion factor for width (pixels to mm?)
POSITIONING_SCALE = 3000  # Scale factor for positioning calculation


# --- Utility Functions ---
def select_folder():
    """Prompts the user to select a folder containing the row profile files."""
    Tk().withdraw()  # Hide the root window
    folder_path = askdirectory(title="Select Folder with Row Profile Files")
    if not folder_path:
        print("No folder selected. Exiting.")
        return None
    return folder_path


def read_profile_data(file_path):
    """
    Reads data from a file and returns it as a list of tuples (x, y).
    Assumes the file contains space-separated values in each row.
    """
    try:
        with open(file_path, "r") as f:
            data = [tuple(map(float, line.split())) for line in f if line.strip()]
        if not data:
            print(f"Warning: File '{file_path}' is empty.")
            return None
        return data
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None


def process_file(file_path, show_plots=False):
    """Processes a single row profile file."""
    data = read_profile_data(file_path)
    if data is None or len(data) < 10:
        print(f"Skipping file {os.path.basename(file_path)} due to insufficient data.")
        return None

    # Unpack x and y values
    x, y = zip(*data)
    x = np.array(x)
    y = np.array(y)

    # Find valley and fit parabola
    parabola_params, vertex_x, vertex_y = find_valley_and_fit_parabola(x, y)
    if vertex_x is None:
        print("Skipping further analysis as parabola fit failed.")
        return None

    # Split data at the vertex
    split_index = int(vertex_x)
    first_half_x, first_half_y = x[:split_index - 6], y[:split_index - 6]
    second_half_x, second_half_y = x[split_index + 6:], y[split_index + 6:]

    # Find the most horizontal segments in each half
    best_first_x, best_first_y = find_horizontal_segment(first_half_x, first_half_y)
    best_second_x, best_second_y = find_horizontal_segment(second_half_x, second_half_y)

    # Find the greatest positive and negative gradient segments in each half
    pos_first_x, pos_first_y, _, pos_slope_first, pos_intercept_first, \
    neg_first_x, neg_first_y, _, neg_slope_first, neg_intercept_first = find_greatest_gradient_segments(first_half_x, first_half_y)

    pos_second_x, pos_second_y, _, pos_slope_second, pos_intercept_second, \
    neg_second_x, neg_second_y, _, neg_slope_second, neg_intercept_second = find_greatest_gradient_segments(second_half_x, second_half_y)

    # Find intersection points for positive and negative gradients
    intersection_pos_first = find_intersection_with_horizontal(pos_slope_first, pos_intercept_first, best_first_y)
    intersection_neg_first = find_intersection_with_horizontal(neg_slope_first, neg_intercept_first, best_first_y)
    intersection_pos_second = find_intersection_with_horizontal(pos_slope_second, pos_intercept_second, best_second_y)
    intersection_neg_second = find_intersection_with_horizontal(neg_slope_second, neg_intercept_second, best_second_y)

    # Calculate outer and inner distances
    outer_left, outer_right = outer_distance(intersection_pos_first, intersection_neg_second, vertex_x)
    inner_left, inner_right = inner_distance(intersection_neg_first, intersection_pos_second, vertex_x)

    # Calculate the average distance
    distance = (outer_left + inner_left) / 2

    # Plot results if show_plots is True
    if show_plots:
        plot_full_analysis(x, y, parabola_params, vertex_x, vertex_y, os.path.basename(file_path),
                           pos_first_x, pos_first_y, neg_first_x, neg_first_y,
                           pos_second_x, pos_second_y, neg_second_x, neg_second_y,
                           pos_slope_first, pos_intercept_first, neg_slope_first, neg_intercept_first,
                           pos_slope_second, pos_intercept_second, neg_slope_second, neg_intercept_second,
                           best_first_x, best_first_y, best_second_x, best_second_y)

    return parabola_params, vertex_x, vertex_y, distance


if __name__ == "__main__":
    try:
        # Prompt the user to select a folder
        folder_path = select_folder()
        if not folder_path:
            exit()

        # Find all row profile files in the selected folder
        file_pattern = os.path.join(folder_path, '*Profile.txt')
        files = glob.glob(file_pattern)

        if not files:
            print("No row profile files found in the selected folder.")
            exit()

        print(f"Distances for folder {folder_path}.")

        # Group files by the first digit of the three-digit number in their names
        grouped_files = {1: [], 2: [], 3: [], 4: []}
        for file in files:
            match = re.search(r'\b(\d{3})\b', os.path.basename(file))
            if match:
                first_digit = int(match.group(1)[0])
                if first_digit in grouped_files:
                    grouped_files[first_digit].append(file)

        # Process each group of files
        for group, group_files in grouped_files.items():
            if not group_files:
                continue

            print(f"\n--- Processing Group {group} ---")
            distances = []

            for file in group_files:
                result = process_file(file)
                if result is not None:
                    _, _, _, distance = result
                    distances.append(distance)

            # Calculate and print the average distance for the group
            if distances:
                average_distance = POSITIONING_SCALE * (np.mean(distances) - 0.5)
                direction = "right" if average_distance > 0 else "left"
                print(f"Average Distance for Group {group}: {abs(average_distance):.4f}um ({direction})")
            else:
                print(f"No valid distances found for Group {group}.")

    except Exception as e:
        print(f"An error occurred: {e}")
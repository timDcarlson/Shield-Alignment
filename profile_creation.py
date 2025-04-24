import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory

# Prompt the user to select a subfolder
def select_subfolder():
    """Prompt the user to select a subfolder for image processing."""
    Tk().withdraw()  # Hide the root window
    folder_path = askdirectory(title="Select a Subfolder")
    if not folder_path:
        raise ValueError("No folder selected. Exiting.")
    return folder_path

# Load the pattern image
def load_pattern(script_dir):
    """Load the pattern image from the script's directory."""
    pattern_path = os.path.join(script_dir, 'pattern.tif')
    if not os.path.exists(pattern_path):
        raise FileNotFoundError(f"Pattern file 'pattern.tif' not found in the script's directory!")
    pattern = cv2.imread(pattern_path, cv2.IMREAD_UNCHANGED)
    if pattern is None:
        raise ValueError("Failed to load the pattern image!")
    return pattern

def plot_and_save(row_profile, cropped_image, image_file, image_dir):
    """Plot the row profile and cropped image, and save the visualization."""
    plt.figure(figsize=(12, 6))

    # Row profile (top)
    plt.subplot(2, 1, 1)
    plt.plot(row_profile, label='Row Profile', color='blue')
    plt.title(f'Row Profile - {image_file}')
    plt.xlabel('Column Index')
    plt.ylabel('Sum of Pixel Intensities')
    plt.legend()
    plt.grid()

    # Cropped image (bottom)
    plt.subplot(2, 1, 2)
    plt.imshow(cropped_image, cmap='gray')
    plt.title('Cropped Image')
    plt.axis('off')

    # Adjust layout and save the plot
    plt.tight_layout()
    output_plot_name = os.path.join(image_dir, f"{os.path.splitext(image_file)[0]}_plot.jpg")
    plt.savefig(output_plot_name, format="jpg", dpi=300)
    plt.close()

def process_images(image_dir, pattern, save_matched=False, create_plot=False):
    """Process each image in the selected folder."""
    # Define the pattern search region of interest (ROI)
    PATTERN_ROW_START, PATTERN_ROW_END = 1040, 1160
    PATTERN_COLUMN_START, PATTERN_COLUMN_END = 908, 1129

    # Get all PNG files in the selected subfolder
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]

    for image_file in image_files:
        try:
            # Construct the full path to the image file
            image_path = os.path.join(image_dir, image_file)

            # Read the 16-bit grayscale image
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Failed to load image: {image_file}")
                continue

            # Crop the pattern search region of interest (ROI)
            pattern_search_roi = image[PATTERN_ROW_START:PATTERN_ROW_END, PATTERN_COLUMN_START:PATTERN_COLUMN_END]

            # Perform template matching
            result = cv2.matchTemplate(pattern_search_roi, pattern, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # Get the top-left and bottom-right corners of the matched region
            top_left = (PATTERN_COLUMN_START + max_loc[0], PATTERN_ROW_START + max_loc[1])
            bottom_right = (top_left[0] + pattern.shape[1], top_left[1] + pattern.shape[0])

            # Define ROW_START, ROW_END, COLUMN_START, COLUMN_END based on the matched pattern
            ROW_START, ROW_END = top_left[1], bottom_right[1]
            COLUMN_START, COLUMN_END = top_left[0], bottom_right[0]

            # Optionally draw a rectangle around the matched region and save the matched image
            if save_matched:
                matched_image = image.copy()
                cv2.rectangle(matched_image, top_left, bottom_right, (255, 0, 0), 2)
                matched_output_path = os.path.join(image_dir, f"{os.path.splitext(image_file)[0]}_matched.jpg")
                cv2.imwrite(matched_output_path, matched_image)

            # Crop the region of interest (ROI) based on the matched pattern
            cropped_image = image[ROW_START:ROW_END, COLUMN_START:COLUMN_END]

            # Collapse all columns to compute the row profile
            row_profile = np.sum(cropped_image, axis=0)  # Sum of each column
            output_profile_path = os.path.join(image_dir, f'{image_file} - Row Profile.txt')
            np.savetxt(output_profile_path, np.column_stack((np.arange(len(row_profile)), row_profile)), fmt='%d')

            # Plot and save the visualization if the plot option is enabled
            if create_plot:
                plot_and_save(row_profile, cropped_image, image_file, image_dir)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# Main script execution
if __name__ == "__main__":
    # Get the path to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the pattern image
    pattern = load_pattern(script_dir)

    # Prompt the user to select a subfolder for image processing
    image_dir = select_subfolder()

    # Process the images with options to enable or disable image outputs
    process_images(image_dir, pattern, save_matched=False, create_plot=False) # Set flags to True to enable image outputs
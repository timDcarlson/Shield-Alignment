import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# --- Configuration ---
VALLEY_PROMINENCE = 0.5
PEAK_PROMINENCE = 100
PARABOLA_FIT_RANGE = 3  # Points +/- from the lowest valley index



# --- Core Analysis Functions ---

def parabola(x, a, b, c):
    """Defines a parabolic function."""
    return a * x**2 + b * x + c

def fit_parabola(x, y):
    """
    Fits a parabola (y = ax^2 + bx + c) to the data using scipy.optimize.curve_fit.
    """
    try:
        params, _ = curve_fit(parabola, x, y)
        a, b, c = params
        vertex_x = -b / (2 * a)
        vertex_y = parabola(vertex_x, a, b, c)
        return params, vertex_x, vertex_y
    except Exception as e:
        print(f"Error fitting parabola: {e}")
        return None, None, None

def find_valley_and_fit_parabola(x, data, fit_range=PARABOLA_FIT_RANGE, prominence=VALLEY_PROMINENCE):
    """Finds the lowest valley and fits a parabola around it."""
    valleys, _ = find_peaks(-data, prominence=prominence)
    if len(valleys) == 0:
        print("Warning: No valleys found. Using global minimum.")
        lowest_valley_index = np.argmin(data)
    else:
        lowest_valley_index = valleys[np.argmin(data[valleys])]

    fit_start = max(0, lowest_valley_index - fit_range)
    fit_end = min(len(data), lowest_valley_index + fit_range + 1)
    fit_x = x[fit_start:fit_end]
    fit_y = data[fit_start:fit_end]

    if len(fit_x) < 3:
        print("Warning: Not enough points to fit a parabola.")
        return None, None, None

    return fit_parabola(fit_x, fit_y)


def find_horizontal_segment(x, y, segment_size=9):
    """
    Finds the set of points in the data that most closely approximates a horizontal line
    and ensures the segment contains the maximum y value.

    Args:
        x (np.ndarray): The x-coordinates of the data.
        y (np.ndarray): The y-coordinates of the data.
        segment_size (int): The number of consecutive points to consider.

    Returns:
        tuple: (best_x, best_y) The x and y values of the most horizontal segment.
    """
    max_y_index = np.argmax(y)  # Index of the maximum y value
    min_slope = float('inf')
    best_x = None
    best_y = None

    # Iterate through all possible segments
    for i in range(len(x) - segment_size + 1):
        segment_x = x[i:i + segment_size]
        segment_y = y[i:i + segment_size]

        # Check if the segment contains the maximum y value
        if max_y_index not in range(i, i + segment_size):
            continue  # Skip this segment if it doesn't include the max y value

        # Fit a line to the segment
        slope, _ = np.polyfit(segment_x, segment_y, 1)

        # Check if this segment is more horizontal
        if abs(slope) < min_slope:
            min_slope = abs(slope)
            best_x = segment_x
            best_y = segment_y

    return best_x, best_y

def find_greatest_gradient_segments(x, y, segment_size=3):
    """
    Finds the segments in the data with the greatest positive and negative gradients
    and calculates the best-fit lines for those segments.

    Args:
        x (np.ndarray): The x-coordinates of the data.
        y (np.ndarray): The y-coordinates of the data.
        segment_size (int): The number of consecutive points to consider (default is 3).

    Returns:
        tuple: (pos_x, pos_y, max_positive_gradient, pos_slope, pos_intercept,
                neg_x, neg_y, max_negative_gradient, neg_slope, neg_intercept)
               The x and y values of the segments with the greatest positive and negative gradients,
               their corresponding gradient values, and the slope and intercept of the best-fit lines.
    """
    if len(x) < segment_size:
        print("Error: Not enough points in the dataset to find gradients.")
        return None, None, None, None, None, None, None, None, None, None

    max_positive_gradient = float('-inf')
    max_negative_gradient = float('inf')
    pos_x = None
    pos_y = None
    neg_x = None
    neg_y = None
    pos_slope = None
    pos_intercept = None
    neg_slope = None
    neg_intercept = None

    # Iterate through all possible segments
    for i in range(len(x) - segment_size + 1):
        segment_x = x[i:i + segment_size]
        segment_y = y[i:i + segment_size]

        # Fit a line to the segment
        slope, intercept = np.polyfit(segment_x, segment_y, 1)

        # Check for greatest positive gradient
        if slope > max_positive_gradient:
            max_positive_gradient = slope
            pos_x = segment_x
            pos_y = segment_y
            pos_slope = slope
            pos_intercept = intercept

        # Check for greatest negative gradient
        if slope < max_negative_gradient:
            max_negative_gradient = slope
            neg_x = segment_x
            neg_y = segment_y
            neg_slope = slope
            neg_intercept = intercept

    return pos_x, pos_y, max_positive_gradient, pos_slope, pos_intercept, \
           neg_x, neg_y, max_negative_gradient, neg_slope, neg_intercept

def find_intersection_with_horizontal(pos_slope, pos_intercept, horizontal_y_values):
    """
    Finds the intersection between the best-fit line for the max positive gradient
    and the average height of the horizontal line data points.

    Args:
        pos_slope (float): Slope of the best-fit line for the max positive gradient.
        pos_intercept (float): Intercept of the best-fit line for the max positive gradient.
        horizontal_y_values (np.ndarray): Y-values of the horizontal line data points.

    Returns:
        float: The x-coordinate of the intersection point.
    """
    if horizontal_y_values is None or len(horizontal_y_values) == 0:
        print("Error: Horizontal line data points are missing or empty.")
        return None

    # Calculate the average height of the horizontal line
    horizontal_y_avg = np.mean(horizontal_y_values)

    # Solve for x in the line equation: y = mx + b
    # horizontal_y_avg = pos_slope * x + pos_intercept
    # => x = (horizontal_y_avg - pos_intercept) / pos_slope
    if pos_slope == 0:
        print("Error: Slope of the best-fit line is zero, no intersection.")
        return None

    intersection_x = (horizontal_y_avg - pos_intercept) / pos_slope
    return intersection_x


def outer_distance(intersection_pos_first, intersection_neg_second, vertex_x):
    """
    Calculates the differences between:
    - intersection_pos_first and vertex_x
    - intersection_neg_second and vertex_x

    Args:
        intersection_pos_first (float): The x-coordinate of the intersection with the positive gradient in the first half.
        intersection_neg_second (float): The x-coordinate of the intersection with the negative gradient in the second half.
        vertex_x (float): The x-coordinate of the vertex.

    Returns:
        tuple: (distance_pos, distance_neg)
               distance_pos is the difference between intersection_pos_first and vertex_x.
               distance_neg is the difference between intersection_neg_second and vertex_x.
    """
    if intersection_pos_first is None or intersection_neg_second is None or vertex_x is None:
        print("Error: One or more inputs to outer_distance are None.")
        return None, None

    # Calculate the distances
    distance_left = abs(intersection_pos_first - vertex_x)
    distance_right = abs(intersection_neg_second - vertex_x)

    rel_distance_left = distance_left /abs(intersection_pos_first - intersection_neg_second) if intersection_pos_first != intersection_neg_second else 0
    rel_distance_right = distance_right /abs(intersection_pos_first - intersection_neg_second) if intersection_pos_first != intersection_neg_second else 0

    return rel_distance_left, rel_distance_right


def inner_distance(intersection_pos_second, intersection_neg_first, vertex_x):
    """
    Calculates the differences between:
    - intersection_pos_first and vertex_x
    - intersection_neg_second and vertex_x

    Args:
        intersection_pos_first (float): The x-coordinate of the intersection with the positive gradient in the first half.
        intersection_neg_second (float): The x-coordinate of the intersection with the negative gradient in the second half.
        vertex_x (float): The x-coordinate of the vertex.

    Returns:
        tuple: (distance_pos, distance_neg)
               distance_pos is the difference between intersection_pos_first and vertex_x.
               distance_neg is the difference between intersection_neg_second and vertex_x.
    """
    if intersection_pos_second is None or intersection_neg_first is None or vertex_x is None:
        print("Error: One or more inputs to outer_distance are None.")
        return None, None

    # Calculate the distances
    distance_right = abs(intersection_pos_second - vertex_x)
    distance_left = abs(intersection_neg_first - vertex_x)

    rel_distance_left = distance_left /abs(intersection_pos_second - intersection_neg_first) if intersection_pos_second != intersection_neg_first else 0
    rel_distance_right = distance_right /abs(intersection_pos_second - intersection_neg_first) if intersection_pos_second != intersection_neg_first else 0

    return rel_distance_left, rel_distance_right
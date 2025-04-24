# Shield Alignment – Row Profile Analysis

## Overview
This repository provides Python utilities for analyzing row profile data extracted from images of cartridge alignment within a shield. Using a combination of peak detection, gradient analysis, and position-based metrics, it calculates relative cartridge positions.

## Features
- **Automated Folder Selection**: Uses a file dialog for selecting profile data folders.
- **Peak & Valley Detection**: Identifies key points using prominence filtering.
- **Parabolic Fitting**: Models valley points to refine alignment estimations.
- **Gradient-Based Feature Extraction**: Identifies regions with maximal slopes.
- **Intersection Calculations**: Computes positional references for alignment analysis.
- **Distance Metrics**: Computes inner and outer distance ratios for alignment assessment.
- **File Grouping & Aggregation**: Automatically organizes files based on identifiers.

## Installation
Ensure required dependencies are installed:
```bash
pip install numpy scipy opencv-python matplotlib
```
Then, clone the repository:
```bash
git clone https://github.com/timDcarlson/Shield-Alignment.git
cd Shield-Alignment
```

## Usage
Example usage for processing a folder of row profile files:
```python
from profile_analysis import process_file

file_path = "path/to/Profile.txt"
result = process_file(file_path, show_plots=True)
print("Alignment Results:", result)
```
This will compute key positional metrics and optionally generate visualization plots.

## Contributing
Contributions are welcome! Submit issues or pull requests for improvements.

## Acknowledgments
Special thanks to contributors and resources that supported this project.

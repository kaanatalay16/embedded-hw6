"""
Quick training script for Handwritten Digit Recognition
EE 4065 - Embedded Digital Image Processing
Kaan Atalay - 150720057

Run this script from the project root directory.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Change to src directory for relative imports
os.chdir(os.path.join(os.path.dirname(__file__), 'src'))

# Run training
from train import main

if __name__ == "__main__":
    results = main()
    print("\nTraining completed successfully!")
    print("Check the 'results' folder for outputs.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import tkinter as tk
from tkinter import filedialog

# Function to select file (Mac and Windows compatible)
def get_file_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Cleaned Data CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

# Function to perform regression and calculate metrics
def perform_regression(df):
    x = df['Pretreatment Stretch Percent']
    y = df['Estimated Shrinkage X']

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    results = {
        "Intercept (β0)": intercept,
        "Slope (β1)": slope,
        "R-squared": r_value**2,
        "P-value (for slope)": p_value
    }
    return results, slope, intercept

# Function to plot scatter + regression line + residual plot
def plot_regression_with_residuals(df, intercept, slope):
    x = df['Pretreatment Stretch Percent']
    y = df['Estimated Shrinkage X']
    predicted_y = intercept + slope * x
    residuals = y - predicted_y

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot with regression line
    ax[0].scatter(x, y, color='blue', edgecolor='k', alpha=0.6, label="Observed Data")
    ax[0].plot(x, predicted_y, color='red', linewidth=2, label=f"y = {slope:.2f}x + {intercept:.2f}")
    ax[0].set_title('Regression: Pretreatment Stretch vs Shrinkage X')
    ax[0].set_xlabel('Pretreatment Stretch Percent')
    ax[0].set_ylabel('Estimated Shrinkage X (%)')
    ax[0].legend()
    ax[0].grid(True)

    # Residual plot
    ax[1].scatter(x, residuals, color='orange', edgecolor='k', alpha=0.6)
    ax[1].axhline(0, color='red', linestyle='--', linewidth=1)
    ax[1].set_title('Residual Plot')
    ax[1].set_xlabel('Pretreatment Stretch Percent')
    ax[1].set_ylabel('Residual (Actual - Predicted)')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

# Function to save regression results to CSV
def save_results_to_csv(results):
    save_path = filedialog.asksaveasfilename(
        title="Save Regression Results As",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv")]
    )
    if save_path:
        results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
        results_df.to_csv(save_path, index=False)
        print(f"Results saved to: {save_path}")
    else:
        print("Results not saved.")

# Main function
def main():
    file_path = get_file_path()
    if not file_path:
        print("No file selected. Exiting.")
        return

    # Load data
    df = pd.read_csv(file_path)

    # Ensure numeric types (just in case)
    df['Estimated Shrinkage X'] = pd.to_numeric(df['Estimated Shrinkage X'], errors='coerce')
    df['Pretreatment Stretch Percent'] = pd.to_numeric(df['Pretreatment Stretch Percent'], errors='coerce')
    df = df.dropna(subset=['Estimated Shrinkage X', 'Pretreatment Stretch Percent'])

    # Perform regression
    results, slope, intercept = perform_regression(df)

    # Print results in terminal
    print("\n=========================================")
    print("Linear Regression Analysis: Pretreatment Stretch vs Estimated Shrinkage X")
    print("=========================================")
    for key, value in results.items():
        if "P-value" in key:
            print(f"{key}: {value:.2e}")  # Scientific notation for p-value
        else:
            print(f"{key}: {value:.6f}")
    print("\nInterpretation:")
    print(f"- Baseline shrinkage (0% stretch) = {results['Intercept (β0)']:.2f}%")
    print(f"- Each 1% pretreatment stretch changes shrinkage by {results['Slope (β1)']:.2f}%")
    print(f"- R-squared: {results['R-squared']:.3f} (Stretch explains {results['R-squared']:.1%} of shrinkage variation)")
    if results['P-value (for slope)'] < 0.05:
        print("- This relationship is statistically significant.")
    else:
        print("- This relationship is not statistically significant.")
    
    # Plot regression and residual plots
    plot_regression_with_residuals(df, intercept, slope)

    # Offer to save results
    save_results_to_csv(results)

if __name__ == "__main__":
    main()

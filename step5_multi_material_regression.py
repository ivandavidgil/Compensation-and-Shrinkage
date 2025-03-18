import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_pdf import PdfPages

# Function to select file
def get_file_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Cleaned Data CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

# Function to run regression on a subset (single material)
def run_regression(df):
    x = df['Pretreatment Stretch Percent']
    y = df['Estimated Shrinkage X']
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return intercept, slope, r_value**2, p_value

# Function to plot and save per-material scatter + residual plot
def plot_material_regression(df, material, intercept, slope, pdf_pages):
    x = df['Pretreatment Stretch Percent']
    y = df['Estimated Shrinkage X']
    predicted_y = intercept + slope * x
    residuals = y - predicted_y

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter + Regression Line
    ax[0].scatter(x, y, color='blue', edgecolor='k', alpha=0.6, label='Observed Data')
    ax[0].plot(x, predicted_y, color='red', linewidth=2, label=f"y = {slope:.2f}x + {intercept:.2f}")
    ax[0].set_title(f'{material} - Regression')
    ax[0].set_xlabel('Pretreatment Stretch Percent')
    ax[0].set_ylabel('Estimated Shrinkage X (%)')
    ax[0].legend()
    ax[0].grid(True)

    # Residual Plot
    ax[1].scatter(x, residuals, color='orange', edgecolor='k', alpha=0.6)
    ax[1].axhline(0, color='red', linestyle='--', linewidth=1)
    ax[1].set_title(f'{material} - Residuals')
    ax[1].set_xlabel('Pretreatment Stretch Percent')
    ax[1].set_ylabel('Residual (Actual - Predicted)')
    ax[1].grid(True)

    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)

# Main process
def main():
    file_path = get_file_path()
    if not file_path:
        print("No file selected. Exiting.")
        return

    df = pd.read_csv(file_path)

    # Ensure numeric columns (just in case)
    df['Estimated Shrinkage X'] = pd.to_numeric(df['Estimated Shrinkage X'], errors='coerce')
    df['Pretreatment Stretch Percent'] = pd.to_numeric(df['Pretreatment Stretch Percent'], errors='coerce')

    df = df.dropna(subset=['Estimated Shrinkage X', 'Pretreatment Stretch Percent', 'Material'])

    # Prepare results storage
    summary_rows = []
    all_data_with_predictions = []

    # Open PDF for plots
    pdf_path = filedialog.asksaveasfilename(
        title="Save Regression Plots PDF As",
        defaultextension=".pdf",
        filetypes=[("PDF Files", "*.pdf")]
    )
    pdf_pages = PdfPages(pdf_path)

    # Process each material
    for material, material_df in df.groupby('Material'):
        if len(material_df) < 10:
            print(f"Skipping {material} (too few data points: {len(material_df)})")
            continue

        intercept, slope, r_squared, p_value = run_regression(material_df)

        print(f"Material: {material}")
        print(f"  Rows: {len(material_df)}")
        print(f"  Intercept: {intercept:.3f}")
        print(f"  Slope: {slope:.3f}")
        print(f"  R-squared: {r_squared:.3f}")
        print(f"  P-value: {p_value:.2e}\n")

        # Store summary row
        summary_rows.append({
            'Material': material,
            'Intercept (β0)': intercept,
            'Slope (β1)': slope,
            'R-squared': r_squared,
            'P-value': p_value,
            'Rows Used': len(material_df)
        })

        # Predict shrinkage and store all data for master CSV
        material_df['Predicted Shrinkage X'] = intercept + slope * material_df['Pretreatment Stretch Percent']
        all_data_with_predictions.append(material_df)

        # Plot regression and residuals
        plot_material_regression(material_df, material, intercept, slope, pdf_pages)

    pdf_pages.close()
    print(f"Plots saved to: {pdf_path}")

    # Save summary to CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = filedialog.asksaveasfilename(
        title="Save Regression Summary CSV As",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv")]
    )
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to: {summary_csv_path}")

    # Save master data with predictions
    all_data_df = pd.concat(all_data_with_predictions, ignore_index=True)
    master_csv_path = filedialog.asksaveasfilename(
        title="Save Master Data with Predictions CSV As",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv")]
    )
    all_data_df.to_csv(master_csv_path, index=False)
    print(f"Master data saved to: {master_csv_path}")

if __name__ == "__main__":
    main()

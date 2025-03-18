import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import tkinter as tk
from tkinter import filedialog

def get_file_path():
    """ Open file dialog to select cleaned data file. """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Cleaned Data CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

def calculate_correlations(df):
    """ Calculate Pearson and Spearman correlations. """
    pearson_corr, pearson_p = pearsonr(df['Pretreatment Stretch Percent'], df['Estimated Shrinkage X'])
    spearman_corr, spearman_p = spearmanr(df['Pretreatment Stretch Percent'], df['Estimated Shrinkage X'])
    
    results = {
        "Pearson Correlation Coefficient": pearson_corr,
        "Pearson p-value": pearson_p,
        "Spearman Correlation Coefficient": spearman_corr,
        "Spearman p-value": spearman_p
    }
    
    return results

def plot_scatter(df):
    """ Generate scatter plot of Pretreatment Stretch vs Estimated Shrinkage X. """
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Pretreatment Stretch Percent'], df['Estimated Shrinkage X'], alpha=0.6, edgecolor='k')
    plt.title('Scatter Plot: Pretreatment Stretch vs Estimated Shrinkage X')
    plt.xlabel('Pretreatment Stretch Percent')
    plt.ylabel('Estimated Shrinkage X (%)')
    plt.grid(True)
    plt.show()

def save_results_to_csv(results):
    """ Prompt to save correlation results to CSV file. """
    save_path = filedialog.asksaveasfilename(
        title="Save Correlation Results As",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv")]
    )
    if save_path:
        results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
        results_df.to_csv(save_path, index=False)
        print(f"Results saved to: {save_path}")
    else:
        print("Results not saved.")

def main():
    file_path = get_file_path()
    if not file_path:
        print("No file selected. Exiting.")
        return

    # Load the cleaned data
    df = pd.read_csv(file_path)

    # Ensure numeric types (in case they got lost during save/load)
    df['Estimated Shrinkage X'] = pd.to_numeric(df['Estimated Shrinkage X'], errors='coerce')
    df['Pretreatment Stretch Percent'] = pd.to_numeric(df['Pretreatment Stretch Percent'], errors='coerce')

    # Drop any rows with missing data (just in case)
    df = df.dropna(subset=['Estimated Shrinkage X', 'Pretreatment Stretch Percent'])

    # Perform Correlation Analysis
    results = calculate_correlations(df)

    # Print results to terminal with p-values in scientific notation
    print("\n=========================================")
    print("Correlation Analysis: Pretreatment Stretch vs Shrinkage X")
    print("=========================================")
    for key, value in results.items():
        if "p-value" in key:
            print(f"{key}: {value:.2e}")  # Scientific notation for p-values
        else:
            print(f"{key}: {value:.6f}")

    # Show scatter plot
    plot_scatter(df)

    # Ask to save results
    save_results_to_csv(results)

if __name__ == "__main__":
    main()


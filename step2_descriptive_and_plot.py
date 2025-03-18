import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Function to open file picker dialog on Mac
def get_file_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select Cleaned CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

def main():
    file_path = get_file_path()

    if not file_path:
        print("No file selected. Exiting.")
        return

    # Load the cleaned data
    df = pd.read_csv(file_path)

    # Ensure numeric conversion (in case columns are treated as strings)
    df['Estimated Shrinkage X'] = pd.to_numeric(df['Estimated Shrinkage X'], errors='coerce')
    df['Pretreatment Stretch Percent'] = pd.to_numeric(df['Pretreatment Stretch Percent'], errors='coerce')

    # Basic Descriptive Statistics
    summary_stats = df[['Estimated Shrinkage X', 'Pretreatment Stretch Percent']].describe()
    print("\nDescriptive Statistics:")
    print(summary_stats)

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Pretreatment Stretch Percent'], df['Estimated Shrinkage X'], alpha=0.5, edgecolor='k')
    plt.title('Scatter Plot: Pretreatment Stretch % vs Estimated Shrinkage X %')
    plt.xlabel('Pretreatment Stretch Percent')
    plt.ylabel('Estimated Shrinkage X (%)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

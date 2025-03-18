import pandas as pd
import tkinter as tk
from tkinter import filedialog

# Function to open file picker dialog on Mac
def get_file_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select Color Grid Locations CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

# Function to clean the data
def clean_shrinkage_data(df):
    # Make a copy to avoid altering the original dataframe
    df_cleaned = df.copy()

    # Convert shrinkage and stretch percent columns to numeric, removing '%'
    df_cleaned['Estimated Shrinkage X'] = pd.to_numeric(
        df_cleaned['Estimated Shrinkage X'].str.replace('%', ''), errors='coerce'
    )
    df_cleaned['Pretreatment Stretch Percent'] = pd.to_numeric(
        df_cleaned['Pretreatment Stretch Percent'].str.replace('%', ''), errors='coerce'
    )

    # Initial count
    initial_count = len(df_cleaned)

    # Drop rows with missing shrinkage/stretch values
    df_cleaned = df_cleaned.dropna(subset=['Estimated Shrinkage X', 'Pretreatment Stretch Percent'])

    # Remove rows with shrinkage outside ±30% (extreme outliers)
    df_cleaned = df_cleaned[(df_cleaned['Estimated Shrinkage X'] >= -30) & 
                            (df_cleaned['Estimated Shrinkage X'] <= 30)]

    # Remove rows with pretreatment stretch outside -7% to +35%
    df_cleaned = df_cleaned[(df_cleaned['Pretreatment Stretch Percent'] >= -7) & 
                            (df_cleaned['Pretreatment Stretch Percent'] <= 35)]

    # Report on what was removed
    final_count = len(df_cleaned)
    removed_count = initial_count - final_count

    print(f"Initial Rows: {initial_count}")
    print(f"Rows Removed (Errors/Missing/Outliers/Extreme Stretch): {removed_count}")
    print(f"Rows Remaining: {final_count}")

    return df_cleaned

# Main script flow
if __name__ == "__main__":
    file_path = get_file_path()

    if not file_path:
        print("No file selected. Exiting.")
    else:
        # Load your file
        df = pd.read_csv(file_path)

        # Run cleaning and assign to new dataframe
        df_cleaned = clean_shrinkage_data(df)

        # Save cleaned data if needed
        output_path = filedialog.asksaveasfilename(
            title="Save Cleaned File As",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )

        if output_path:
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned file saved to: {output_path}")
        else:
            print("Cleaned file not saved.")

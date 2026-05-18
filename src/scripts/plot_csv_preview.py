import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse

def plot_first_200_points(csv_path, output_dir):
    """
    Reads a CSV file, plots the first 200 points of the 'power' column,
    and saves the plot to the output directory.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Take the first 200 points
        df_subset = df.head(200)
        
        if df_subset.empty:
            print(f"Error: The file {csv_path} is empty.")
            return

        # Check for 'power' column, otherwise use the first numerical column
        if 'power' in df_subset.columns:
            y_data = df_subset['power']
            y_label = 'Power (W)'
        else:
            # Select the first column that is numeric
            numeric_cols = df_subset.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                y_data = df_subset[numeric_cols[0]]
                y_label = numeric_cols[0]
            else:
                print(f"Error: No numeric data found in {csv_path}.")
                return

        # Use 'timestamp' for x-axis if available, else use index
        if 'timestamp' in df_subset.columns:
            x_data = df_subset['timestamp']
            x_label = 'Timestamp'
        else:
            x_data = df_subset.index
            x_label = 'Index'

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, marker='o', linestyle='-', markersize=2, color='b')
        plt.title(f"First 200 Points of {os.path.basename(csv_path)}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot
        file_name = os.path.splitext(os.path.basename(csv_path))[0] + "_preview.png"
        output_path = os.path.join(output_dir, file_name)
        plt.savefig(output_path)
        plt.close()
        
        print(f"Successfully saved plot to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the first 200 points of a CSV file.")
    parser.add_argument("csv_path", help="Path to the input .csv file")
    parser.add_argument("--output_dir", default="/home/scnu2023024258/data/code/PSLG-NILM/static", 
                        help="Directory to save the output chart")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found at {args.csv_path}")
        sys.exit(1)
        
    plot_first_200_points(args.csv_path, args.output_dir)

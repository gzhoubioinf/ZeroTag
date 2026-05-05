import pandas as pd
import os

# --- Configuration ---
DATA_DIR = '../../data'
OUTPUT_FILE = os.path.join(DATA_DIR, 'strain_names.csv')
# ---------------------

def main():
    """
    Combines the four 384-well source plate files into a single
    master file that includes the source plate number for each strain.
    """
    all_strains = []
    
    for i in range(1, 5):
        filename = f'plate{i}.txt'
        filepath = os.path.join(DATA_DIR, filename)
        
        try:
            df = pd.read_csv(filepath, sep='\t')
            df['Plate'] = i  # Add the plate number
            df = df.rename(columns={'strain': 'ID', 'row': 'Row', 'column': 'Column'})
            all_strains.append(df[['ID', 'Row', 'Column', 'Plate']])
        except FileNotFoundError:
            print(f"Warning: Source file not found: {filepath}")

    if all_strains:
        master_df = pd.concat(all_strains, ignore_index=True)
        master_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully created '{OUTPUT_FILE}' with {len(master_df)} total strains from 4 plates.")
    else:
        print("No plate files were processed. Output file was not created.")

if __name__ == '__main__':
    main()

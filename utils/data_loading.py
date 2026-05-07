import pandas as pd
import streamlit as st

@st.cache_data(show_spinner="Loading IRIS data...")
def get_colony_data(iris_path):
    """
    Robustly parses a single .iris file into a pandas DataFrame.
    This function is adapted from the corrected test_bubble_plot.py script.
    """
    try:
        # The file is consistently tab-separated with a '.' decimal.
        df = pd.read_csv(
            iris_path,
            comment='#',
            sep='\t',
            decimal='.'
        )
        return df
    except FileNotFoundError:
        st.error(f"IRIS file not found at path: {iris_path}")
        return None
    except Exception as e:
        st.error(f"Failed to parse IRIS file. Error: {e}")
        return None


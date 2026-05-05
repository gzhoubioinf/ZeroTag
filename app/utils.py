import streamlit as st
import os
import glob
import re

@st.cache_data
def _parse_filename_details(base_name):
    """
    Robustly parses an IRIS filename to extract condition, plate, and batch numbers.
    Example: Urine-5%-3-1_A.JPG.iris -> ('Urine-5%', 3, 1) (Condition, Plate, Batch)
    """
    pattern = re.compile(r'-(\d+)-(\d+)_([A-Z])\.(JPG\.iris)$')
    match = pattern.search(base_name)

    if match:
        try:
            plate_num = int(match.group(1)) # First number is the plate
            batch_num = int(match.group(2)) # Second number is the batch
            condition = base_name[:match.start()]
            return condition, plate_num, batch_num
        except (ValueError, IndexError):
            return None, None, None
    return None, None, None

@st.cache_data
def get_plate_numbers(directory, condition):
    """Get all unique plate numbers for a given condition from IRIS files."""
    if not condition:
        return []
    plate_numbers = set()
    file_pattern = os.path.join(directory, "*.iris")
    for f in glob.glob(file_pattern):
        base_name = os.path.basename(f)
        cond_from_file, plate_num, _ = _parse_filename_details(base_name)
        if cond_from_file == condition and plate_num is not None:
            plate_numbers.add(plate_num)
    return sorted(list(plate_numbers))

@st.cache_data
def get_conditions(directory):
    """Collect names of conditions from all *.iris files."""
    files = glob.glob(os.path.join(directory, "*.iris"))
    condition_names = set()
    for f in files:
        base_name = os.path.basename(f)
        condition, _, _ = _parse_filename_details(base_name)
        if condition:
            condition_names.add(condition)
    return sorted(list(condition_names))

@st.cache_data
def get_batch_numbers(directory, condition, plate_number):
    """Get all unique batch numbers for a given condition and plate from IRIS files."""
    if not condition or plate_number is None:
        return []
    
    batch_numbers = set()
    file_pattern = os.path.join(directory, f"{condition}-{plate_number}-*.iris")

    for f in glob.glob(file_pattern):
        base_name = os.path.basename(f)
        cond_from_file, plate_num_from_file, batch_num = _parse_filename_details(base_name)
        if cond_from_file == condition and plate_num_from_file == plate_number and batch_num is not None:
            batch_numbers.add(batch_num)
            
    return sorted(list(batch_numbers))

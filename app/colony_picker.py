
import streamlit as st

import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import re
from scipy.stats import percentileofscore, gaussian_kde
import cv2
import numpy as np
from matplotlib.patches import Rectangle

from utils.data_loading import load_csv, load_excel, get_colony_data as get_iris_data
from utils.image_handling import extract_colony, crop_img
from app.utils import get_conditions, get_plate_numbers, get_batch_numbers

@st.cache_data
def get_iris_files(directory):
    """Gets a sorted list of all .iris files in the specified directory."""
    return sorted([os.path.basename(f) for f in glob.glob(os.path.join(directory, "*.iris"))])

@st.cache_data
def load_strain_data(config):
    """Loads strain data from CSV or Excel, caching the result."""
    try:
        return load_excel(config['files']['strain_file'])
    except Exception:
        return load_csv(config['files']['strain_file'])

def generate_bubble_plot(df: pd.DataFrame, size_metric: str, color_metric: str, scale_factor: float, highlight_coords=None):
    """
    Generates and returns a bubble plot figure formatted as a 1536-well plate.
    """
    if df.empty or size_metric not in df.columns or color_metric not in df.columns:
        st.warning(f"Plot generation failed: DataFrame is empty or missing required metric columns. "
                   f"Required: '{size_metric}', '{color_metric}'.")
        return None

    df_clean = df.dropna(subset=['row', 'column', size_metric, color_metric])
    
    if df_clean.empty:
        st.warning("No valid data remains after cleaning for plotting.")
        return None

    # Plate dimensions
    NUM_ROWS = 32
    NUM_COLS = 48

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(22, 15))

    scatter = ax.scatter(
        x=df_clean['column'],
        y=df_clean['row'],
        s=df_clean[size_metric] * scale_factor,
        c=df_clean[color_metric],
        cmap='inferno',
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )

    # --- Highlight selected replicates ---
    if highlight_coords:
        for rep_label, coords in highlight_coords.items():
            row, col = coords['row'], coords['col']
            ax.add_patch(Rectangle(
                (col - 0.5, row - 0.5), 1, 1,
                edgecolor='red',
                facecolor='none',
                lw=4,
                zorder=10  # Ensure it's drawn on top
            ))

    # --- Formatting for 1536-well plate ---
    ax.set_xlim(0.5, NUM_COLS + 0.5)
    ax.set_ylim(NUM_ROWS + 0.5, 0.5)
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xticks(np.arange(1, NUM_COLS + 1, 1))
    ax.set_yticks(np.arange(1, NUM_ROWS + 1, 1))
    ax.tick_params(axis='x', rotation=90, labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    ax.set_xticks(np.arange(0.5, NUM_COLS + 1.5, 1), minor=True)
    ax.set_yticks(np.arange(0.5, NUM_ROWS + 1.5, 1), minor=True)
    
    ax.grid(which='minor', linestyle='--', linewidth=0.5, color='gray')

    ax.set_xlabel("Column", fontsize=25, fontweight='bold')
    ax.set_ylabel("Row", fontsize=25, fontweight='bold')
    ax.set_title(f"Plate Overview | Size: {size_metric} | Color: {color_metric}", fontsize=25, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(color_metric.title(), rotation=270, labelpad=15, fontweight='bold',size=20)
    cbar.ax.tick_params(labelsize=20)

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.7, num=5)
    # The labels from legend_elements are based on the scaled sizes.
    # Reverse the scaling to show the actual colony size values.
    legend_labels = []
    for label in labels:
        # Extract the numeric part of the label, which might be a float
        numeric_part = re.search(r'(\d+\.?\d*)', label)
        if numeric_part:
            scaled_size = float(numeric_part.group(1))
            actual_size = int(scaled_size / scale_factor)
            legend_labels.append(f"{actual_size}")

    # Move legend to the left of the plot
    ax.legend(handles, legend_labels, loc="center right", bbox_to_anchor=(-0.15, 0.5),
              title=size_metric.title(), fontsize=25, title_fontsize=25)
    
    # Use manual subplot adjustment to prevent legend and color bar overlap
    plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.05)
    return fig

def colonypicker(config):
    st.title("ZeroTag")

    # CSS to make the table background transparent and reduce spacing. just for aesthetics
    st.markdown("""
        <style>
            ul {
                margin-top: 0px;
                margin-bottom: 0px;
                padding-left: 20px;
            }
            li {
                margin-bottom: 2px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                background-color: transparent;
            }
            th, td {
                background-color: transparent;
                text-align: left;
                padding: 5px;
                vertical-align: top;
                border: none;
            }
        </style>
        """, unsafe_allow_html=True
    )

    strains_df = load_strain_data(config)

    image_directory = config['directories']['image_directory']
    iris_directory = config['directories']['iris_directory']

    with st.spinner("Loading conditions..."):
        conditions = get_conditions(iris_directory)

    if 'display_data' not in st.session_state:
        st.session_state.display_data = None

    with st.sidebar:
        st.header("Data Source")
        if st.button("🔄 Refresh File Lists"):
            get_conditions.clear()
            get_plate_numbers.clear()
            get_batch_numbers.clear()
            st.success("File lists refreshed!")
            st.rerun()

        st.header("Colony Selection")
        st.selectbox("Select a condition", conditions, key='selected_condition')

        if st.session_state.selected_condition:
            plate_numbers = get_plate_numbers(iris_directory, st.session_state.selected_condition)
            if plate_numbers:
                if 'selected_plate' not in st.session_state or st.session_state.selected_plate not in plate_numbers:
                    st.session_state.selected_plate = plate_numbers[0]
                
                st.selectbox("Select Plate Number", plate_numbers, key='selected_plate')
            else:
                st.warning("No plates found for this condition.")
                if 'selected_plate' in st.session_state:
                    st.session_state.selected_plate = None

            batches = []
            if st.session_state.get('selected_plate'):
                batches = get_batch_numbers(iris_directory, st.session_state.selected_condition, st.session_state.selected_plate)
                if batches and 'selected_batch' not in st.session_state:
                    st.session_state.selected_batch = batches[0]
            
            st.selectbox("Select Batch", batches, key='selected_batch')

        method = st.radio("Select colony extraction method:", ("By Strain Name", "By Row and Column"))

        if method == "By Strain Name":
            strain_name = st.selectbox("Select Strain", strains_df['ID'])
            if strain_name:
                r_c_vals = strains_df.loc[strains_df['ID'] == strain_name, ['Row', 'Column']].values[0]
                st.session_state.row, st.session_state.col = int(r_c_vals[0]), int(r_c_vals[1])
                st.session_state.method = "By Strain Name" # Store method in session state
                st.markdown(f"**Selected Strain:** `{strain_name}` (Row {st.session_state.row}, Col {st.session_state.col})")

        else: # By Row and Column
            st.number_input("Enter 1536-well Row:", min_value=1, max_value=32, value=1, step=1, key='row')
            st.number_input("Enter 1536-well Column:", min_value=1, max_value=48, value=1, step=1, key='col')
            
            r_384 = (st.session_state.row + 1) // 2
            c_384 = (st.session_state.col + 1) // 2
            strain_entry = strains_df[(strains_df['Row'] == r_384) & (strains_df['Column'] == c_384)]
            if not strain_entry.empty:
                strain_id = strain_entry.iloc[0]['ID']
                st.session_state.method = "By Row and Column" # Store method in session state
                st.markdown(f"**Parent Strain:** `{strain_id}` (from 384-well at {r_384}, {c_384})")
        
        all_metrics = [
            'colony size', 'circularity', 'colony color intensity', 'biofilm area size',
            'biofilm color intensity', 'biofilm area ratio', 'size normalized color intensity',
            'mean sampled color intensity', 'average pixel saturation', 'opacity', 'max 10% opacity'
        ]
        default_metrics = ['circularity', 'colony size', 'opacity', 'biofilm color intensity']
        
        selected_metrics = st.multiselect("Select metrics to display:", options=all_metrics, default=default_metrics)

        if st.button("Submit"):
            process_data(strains_df, image_directory, iris_directory, method, selected_metrics, st.session_state.selected_condition, st.session_state.selected_plate, st.session_state.selected_batch)

    if st.session_state.display_data:
        data = st.session_state.display_data
        image_path = data["image_path"]
        replicates_data = data["replicates_data"]
        iris_df = data["iris_df"]
        extracted_images = data["extracted_images"]
        
        st.info(f"Displaying data for plate: **{os.path.basename(image_path)}**")

        with st.expander("Show Detected Grid", expanded=False):
            preview_img_bgr = cv2.imread(image_path)
            if preview_img_bgr is not None:
                preview_img_rgb = cv2.cvtColor(preview_img_bgr, cv2.COLOR_BGR2RGB)
                img_h, img_w, _ = preview_img_rgb.shape
                cell_h, cell_w = img_h / 32, img_w / 48
                for rep_label, coords in replicates_data.items():
                    rep_row, rep_col = coords['row'], coords['col']
                    adj_row, adj_col = rep_row - 1, rep_col - 1
                    box_x1 = int(round(adj_col * cell_w))
                    box_y1 = int(round(adj_row * cell_h))
                    box_x2 = int(round(box_x1 + cell_w))
                    box_y2 = int(round(box_y1 + cell_h))
                    cv2.rectangle(preview_img_rgb, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 5)
                st.image(preview_img_rgb, caption="Detected Grid Area with Replicate Locations", use_container_width=True)
            else:
                st.warning("Could not load preview image for grid detection.")

        # --- Bubble Plot Display ---
        st.subheader("Plate-Wide Bubble Plot")
        with st.expander("Show Bubble Plot", expanded=True):
            if iris_df is not None and not iris_df.empty:
                bubble_plot_fig = generate_bubble_plot(
                    df=iris_df,
                    size_metric='colony size',
                    color_metric='biofilm color intensity',
                    scale_factor=0.05,
                    highlight_coords=replicates_data
                )
                if bubble_plot_fig:
                    st.pyplot(bubble_plot_fig)
                else:
                    st.warning("Could not generate bubble plot. Check metric selections and data file.")
            else:
                st.warning("No plate data available to generate a bubble plot.")

        st.write("---")
        st.subheader("Colony Images and Metrics")

        cols = st.columns(4)
        for i, (rep_label, coords) in enumerate(replicates_data.items()):
            with cols[i]:
                extracted = extracted_images.get(rep_label)
                if extracted is not None:
                    rgb_img = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)
                    st.image(rgb_img, caption=f"Rep {rep_label} (R:{coords['row']}, C:{coords['col']})", use_container_width=True)
                else:
                    st.warning(f"No colony at R:{coords['row']}, C:{coords['col']}")
                
                # Extract metrics
                metric_vals = {metric: None for metric in selected_metrics}
                if iris_df is not None:
                    matching = iris_df[(iris_df['row'] == coords['row']) & (iris_df['column'] == coords['col'])]
                    # --- FIX #1: Corrected indentation below ---
                    if not matching.empty:
                        for metric in selected_metrics:
                            if metric in matching.columns and pd.notna(matching[metric].iloc[0]):
                                metric_vals[metric] = matching[metric].iloc[0]
                
                for metric, value in metric_vals.items():
                    display_name = metric.replace('_', ' ').title()
                    if value is not None:
                        st.write(f"{display_name}: {value:.3f}")
                    else:
                        st.write(f"{display_name}: N/A")
        
        if st.button("Save Replicate Images"):
            output_dir = "saved_replicates"
            os.makedirs(output_dir, exist_ok=True)
            
            if st.session_state.method == "By Strain Name":
                strain_name = st.session_state.strain_name
            else:
                strain_name = f"R{st.session_state.row}C{st.session_state.col}"
            condition_name = st.session_state.selected_condition.replace("/", "_").replace("\\", "_")
            saved_files_count = 0
            for rep_label, img_data in extracted_images.items():
                if img_data is not None:
                    filename = f"{strain_name}_{condition_name}_{rep_label}.png"
                    filepath = os.path.join(output_dir, filename)
                    success = cv2.imwrite(filepath, img_data)
                    if success:
                        saved_files_count += 1
            if saved_files_count > 0:
                st.success(f"Successfully saved {saved_files_count} images to the '{output_dir}' directory.")
            else:
                st.warning("No images were available to save.")

def process_data(strains_df, image_directory, iris_directory, method, selected_metrics, selected_condition, selected_plate, selected_batch):
    
    # --- Image and IRIS file paths are now derived directly from the selection ---
    plate_num = st.session_state.selected_plate
    if selected_batch:
        iris_file_pattern = os.path.join(iris_directory, f"{selected_condition}-{plate_num}-{selected_batch}_*.JPG.iris")
    else:
        iris_file_pattern = os.path.join(iris_directory, f"{selected_condition}-{plate_num}-*_*.JPG.iris")

    possible_iris_files = glob.glob(iris_file_pattern)

    if not possible_iris_files:
        batch_display = selected_batch or '[any]'
        st.warning(f"No IRIS file found for condition '{selected_condition}', plate {plate_num}, and batch {batch_display}.")
        st.session_state.display_data = None
        return
    
    # We now have the definitive iris file, so we can find the matching image
    iris_file_path = possible_iris_files[0]
    
    # Use a flexible search to find the corresponding image file
    core_identifier = os.path.basename(iris_file_path).split('.')[0]
    image_search_pattern = os.path.join(image_directory, f"{core_identifier}*.JPG.grid.jpg")
    possible_images = glob.glob(image_search_pattern)

    if not possible_images:
        st.warning(f"Found IRIS file, but could not find matching image file for: {os.path.basename(iris_file_path)}")
        st.session_state.display_data = None
        return
    
    image_path = possible_images[0]

    plate_img = crop_img(image_path)
    if plate_img is None:
        st.error(f"Could not load the plate image from: {image_path}")
        st.session_state.display_data = None
        return
    
    iris_df = None 
    if os.path.exists(iris_file_path):
        iris_df = get_iris_data(iris_file_path) 
    else:
        st.warning(f"This should not happen, but IRIS file not found at: {iris_file_path}")

    # --- Colony coordinate logic remains the same ---
    if method == "By Strain Name":
        strain_name = st.session_state.get('strain_name')
        if strain_name:
            strain_info = strains_df[strains_df['ID'] == strain_name]
            if strain_info.empty:
                st.error(f"Could not find strain information for ID: {strain_name}")
                return
        else: # Should not happen if UI is working
            return
    else: # By Row and Column
        row_1536, col_1536 = st.session_state.row, st.session_state.col
        row_384 = (row_1536 + 1) // 2
        col_384 = (col_1536 + 1) // 2
        strain_info = strains_df[(strains_df['Row'] == row_384) & (strains_df['Column'] == col_384)]

    if strain_info.empty:
        st.error("Could not find strain information for the selection.")
        st.session_state.display_data = None
        return
    
    r_384, c_384 = strain_info.iloc[0]['Row'], strain_info.iloc[0]['Column']

    r_tl_1536 = (r_384 * 2) - 1
    c_tl_1536 = (c_384 * 2) - 1
    replicates_data = {
        'A': {'row': r_tl_1536, 'col': c_tl_1536},
        'B': {'row': r_tl_1536, 'col': c_tl_1536 + 1},
        'C': {'row': r_tl_1536 + 1, 'col': c_tl_1536 + 1},
        'D': {'row': r_tl_1536 + 1, 'col': c_tl_1536},
    }

    extracted_images = {}
    for rep_label, coords in replicates_data.items():
        rep_row, rep_col = coords['row'], coords['col']
        adj_row, adj_col = rep_row - 1, rep_col - 1
        extracted_images[rep_label] = extract_colony(plate_img, adj_row, adj_col)

    st.session_state.display_data = {
        "image_path": image_path,
        "replicates_data": replicates_data,
        "iris_df": iris_df,
        "extracted_images": extracted_images,
    }

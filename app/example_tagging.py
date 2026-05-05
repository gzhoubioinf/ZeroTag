import streamlit as st
import os
import glob
import pandas as pd
import cv2
import numpy as np

# Assuming these functions are in zerotag or a shared utils file
from app.utils import get_conditions, get_plate_numbers, get_batch_numbers
from utils.data_loading import get_colony_data as get_iris_data
from utils.image_handling import extract_colony, find_grid_by_cell_contours, crop_img

def make_square(image, size=150):
    """Resizes an image to a square, padding if necessary to maintain aspect ratio."""
    if image is None:
        return np.zeros((size, size, 3), dtype=np.uint8) 
        
    h, w, _ = image.shape
    if h == w:
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    
    max_dim = max(h, w)
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left
    
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return cv2.resize(padded_image, (size, size), interpolation=cv2.INTER_AREA)

@st.cache_data
def load_example_images(directory):
    """
    Loads example images from a directory and returns them in a specific, predefined order.
    """
    # Define the desired order and prefixes
    desired_order = ["No Growth", "Poor Growth", "Normal Growth", "Over Growth"]
    
    # Load all images into a dictionary for quick lookup
    all_images = {}
    valid_extensions = ("*.png", "*.jpg", "*.jpeg")
    for ext in valid_extensions:
        for img_path in glob.glob(os.path.join(directory, ext)):
            image_name = os.path.splitext(os.path.basename(img_path))[0]
            image = cv2.imread(img_path)
            if image is not None:
                square_image = make_square(image, size=150)
                all_images[image_name] = cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB)
    
    # Build the final list in the desired order
    ordered_examples = []
    for name in desired_order:
        if name in all_images:
            ordered_examples.append((name, all_images[name]))
            
    return ordered_examples

def update_plate_selection():
    """Callback to update the plate selection when the condition changes."""
    # This function is called AFTER st.session_state.selected_condition has been updated by the selectbox
    iris_directory = st.session_state.config['directories']['iris_directory']
    plate_numbers = get_plate_numbers(iris_directory, st.session_state.selected_condition)
    
    # When condition changes, reset everything to start fresh
    if plate_numbers:
        st.session_state.selected_plate = plate_numbers[0]
    else:
        st.session_state.selected_plate = None
    
    # Crucially, reset all other dependent states
    st.session_state.selected_batch = None
    st.session_state.zero_colonies = None
    st.session_state.current_colony_idx = 0
    st.session_state.annotations = {}

def go_to_next_plate():
    """Callback to advance to the next plate."""
    image_directory = st.session_state.config['directories']['image_directory']
    plate_numbers = get_plate_numbers(image_directory, st.session_state.selected_condition)
    current_plate_index = plate_numbers.index(st.session_state.selected_plate)
    
    if current_plate_index < len(plate_numbers) - 1:
        next_plate = plate_numbers[current_plate_index + 1]
        st.session_state.selected_plate = next_plate
        # Reset state for the new plate
        st.session_state.zero_colonies = None
        st.session_state.current_colony_idx = 0
        st.session_state.annotations = {}
    else:
        # This was the last plate
        st.session_state.finished_condition = True


def example_tagging_app(config):
    st.title("Tag Zero-Size Colonies")
    
    # --- Initialize session state variables ---
    if 'current_colony_idx' not in st.session_state:
        st.session_state.current_colony_idx = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}
    if 'plate_finished' not in st.session_state:
        st.session_state.plate_finished = False
    if 'go_to_next_plate' not in st.session_state:
        st.session_state.go_to_next_plate = False
    if 'go_to_previous_plate' not in st.session_state:
        st.session_state.go_to_previous_plate = False

    # Store config in session state to make it accessible in callbacks
    st.session_state.config = config

    image_directory = config['directories']['image_directory']
    iris_directory = config['directories']['iris_directory']
    example_directory = "data/Example_images"
    annotations_directory = "annotations"
    os.makedirs(annotations_directory, exist_ok=True)

    # --- Create a unified list of all plates and batches for seamless navigation ---
    conditions = get_conditions(iris_directory)
    all_selections = []
    if conditions:
        for condition in conditions:
            plates = get_plate_numbers(iris_directory, condition)
            for plate in plates:
                batches = get_batch_numbers(iris_directory, condition, plate)
                for batch in batches:
                    all_selections.append({'condition': condition, 'plate': plate, 'batch': batch})

    # --- Handle Inter-Plate Navigation ---
    if all_selections and 'selected_condition' in st.session_state and 'selected_plate' in st.session_state and 'selected_batch' in st.session_state:
        current_selection = {
            'condition': st.session_state.selected_condition, 
            'plate': st.session_state.selected_plate,
            'batch': st.session_state.selected_batch
        }
        
        try:
            current_index = all_selections.index(current_selection)

            if st.session_state.go_to_next_plate:
                st.session_state.go_to_next_plate = False
                if current_index < len(all_selections) - 1:
                    next_item = all_selections[current_index + 1]
                    st.session_state.selected_condition = next_item['condition']
                    st.session_state.selected_plate = next_item['plate']
                    st.session_state.selected_batch = next_item['batch']
                    st.session_state.zero_colonies = None
                    st.session_state.current_colony_idx = 0
                    st.session_state.annotations = {}
                    st.rerun()
                else:
                    st.info("This is the last batch of the last plate of the last condition.")

            if st.session_state.go_to_previous_plate:
                st.session_state.go_to_previous_plate = False
                if current_index > 0:
                    prev_item = all_selections[current_index - 1]
                    st.session_state.selected_condition = prev_item['condition']
                    st.session_state.selected_plate = prev_item['plate']
                    st.session_state.selected_batch = prev_item['batch']
                    st.session_state.zero_colonies = None
                    st.session_state.current_colony_idx = 0
                    st.session_state.annotations = {}
                    st.rerun()
                else:
                    st.info("This is the first batch of the first plate of the first condition.")
        except ValueError:
            # Current selection not in list — clear stale navigation flags to avoid phantom jumps
            st.session_state.go_to_next_plate = False
            st.session_state.go_to_previous_plate = False
    
    # --- Handle automatic plate advancement ---
    if st.session_state.get('plate_finished', False):
        st.session_state.plate_finished = False # Reset the flag

        plate_numbers = get_plate_numbers(iris_directory, st.session_state.selected_condition)
        current_plate_index = plate_numbers.index(st.session_state.selected_plate)

        if current_plate_index < len(plate_numbers) - 1:
            next_plate = plate_numbers[current_plate_index + 1]
            st.session_state.selected_plate = next_plate
            st.info(f"Automatically moving to next plate: {next_plate}")
            # Reset state for the new plate
            st.session_state.zero_colonies = None
            st.session_state.current_colony_idx = 0
            st.session_state.annotations = {}
            st.rerun() # Rerun to process the new plate
        else:
            st.success("All plates for this condition have been tagged!")
            st.session_state.zero_colonies = "COMPLETED"

    example_images = load_example_images(example_directory)

    if not example_images:
        st.error("No example images found in `data/Example_images`. Please add some images to proceed.")
        return

    # --- Sidebar for Plate Selection ---
    with st.sidebar:
        st.header("Data Source")
        if st.button("🔄 Refresh File Lists"):
            get_conditions.clear()
            get_plate_numbers.clear()
            get_batch_numbers.clear()
            st.success("File lists refreshed!")
            st.rerun()

        st.header("Plate Selection")
        
        with st.spinner("Loading conditions..."):
            conditions = get_conditions(iris_directory)
        
        if not conditions:
            st.warning("No conditions found in the IRIS directory.")
            return

        if 'selected_condition' not in st.session_state:
            st.session_state.selected_condition = conditions[0] if conditions else None

        st.selectbox("Select a condition", conditions, key='selected_condition', on_change=update_plate_selection)

        plate_numbers = []
        if st.session_state.selected_condition:
            plate_numbers = get_plate_numbers(iris_directory, st.session_state.selected_condition)
            if not plate_numbers:
                st.warning("No plates found for this condition.")
            elif 'selected_plate' not in st.session_state or st.session_state.selected_plate not in plate_numbers:
                st.session_state.selected_plate = plate_numbers[0]

        st.selectbox("Select Plate Number", plate_numbers, key='selected_plate')

        batches = []
        if st.session_state.get('selected_plate'):
            batches = get_batch_numbers(iris_directory, st.session_state.selected_condition, st.session_state.selected_plate)
            if batches and ('selected_batch' not in st.session_state or st.session_state.selected_batch not in batches):
                st.session_state.selected_batch = batches[0]
        
        st.selectbox("Select Batch", batches, key='selected_batch')

        #if st.button("Reprocess Plate"):
            #st.session_state.zero_colonies = None # Reset
            #st.session_state.current_colony_idx = 0
            #st.session_state.annotations = {}
            #st.rerun()
        col1, col2, col3 = st.columns([1, 3, 1])

        with col2: # Place the button in the middle, wider column
            if st.button("🔄 Reprocess Plate", 
                        type="primary", 
                        use_container_width=True): # This makes the button fill the column
                
                st.session_state.zero_colonies = None # Reset
                st.session_state.current_colony_idx = 0
                st.session_state.annotations = {}
                st.rerun()

    # --- Automatic Processing ---
    # If there are no colonies loaded for the current plate, find them.
    if st.session_state.get('zero_colonies') is None:
        selected_plate = st.session_state.get('selected_plate')
        selected_condition = st.session_state.get('selected_condition')
        selected_batch = st.session_state.get('selected_batch')

        if selected_plate and selected_condition and selected_batch:
            # Find the exact IRIS file and corresponding image file
            iris_file_pattern = os.path.join(iris_directory, f"{selected_condition}-{selected_plate}-{selected_batch}_*.JPG.iris")
            possible_iris_files = glob.glob(iris_file_pattern)

            if not possible_iris_files:
                st.error(f"Could not find a matching IRIS file for the selection.")
                st.stop()
            
            iris_file_path = possible_iris_files[0]
            base_name = os.path.basename(iris_file_path).replace('.JPG.iris', '')
            
            image_path_pattern = os.path.join(image_directory, f"{base_name}.JPG.grid.jpg")
            possible_images = glob.glob(image_path_pattern)
            
            if not possible_images:
                st.error(f"Could not find a matching image file for '{os.path.basename(iris_file_path)}'.")
                st.stop()
            
            image_path = possible_images[0]
            
            plate_img = crop_img(image_path)
            if plate_img is None:
                st.error(f"Failed to load or crop image: {image_path}")
                st.stop()
            
            iris_df = get_iris_data(iris_file_path)
            if iris_df is not None:
                zero_colonies = iris_df[iris_df['colony size'] == 0]
                if not zero_colonies.empty:
                    st.session_state.plate_image_path = image_path
                    st.session_state.plate_image = plate_img
                    st.session_state.zero_colonies = zero_colonies.to_dict('records')
                    st.success(f"Found {len(zero_colonies)} colonies with size 0.")
                    st.rerun()
                else:
                    st.info("No colonies with size 0 found on this plate.")
                    st.session_state.zero_colonies = "COMPLETED"
            else:
                st.error("Failed to load IRIS data.")
                st.session_state.zero_colonies = "COMPLETED"


    # --- Main Display for Tagging ---
    if 'zero_colonies' in st.session_state and isinstance(st.session_state.zero_colonies, list) and st.session_state.zero_colonies:
        
        idx = st.session_state.current_colony_idx
        colony = st.session_state.zero_colonies[idx]
        row, col = int(colony['row']), int(colony['column'])

        st.header(f"Tagging Colony {idx + 1} of {len(st.session_state.zero_colonies)}")
        st.write(f"**Location:** Row `{row}`, Column `{col}`")

        with st.expander("Show Detected Grid on Plate", expanded=False):
            preview_img = st.session_state.plate_image 
            if preview_img is not None:
                preview_img_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                
                # Use the robust grid detection to find the colony's true position
                grid_origin, cell_size = find_grid_by_cell_contours(preview_img)
                grid_offset_x, grid_offset_y = grid_origin
                cell_w, cell_h = cell_size

                # Calculate the top-left corner of the target cell, accounting for grid offset
                box_x1 = grid_offset_x + ((col - 1) * cell_w)
                box_y1 = grid_offset_y + ((row - 1) * cell_h)
                box_x2 = box_x1 + cell_w
                box_y2 = box_y1 + cell_h
                
                cv2.rectangle(preview_img_rgb, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 5)
                
                st.image(preview_img_rgb, caption="Plate Image with Current Colony Highlighted", use_container_width=True)
            else:
                st.warning("Could not load preview image for grid detection.")

        # Extract and display the colony image — centred
        colony_img = extract_colony(st.session_state.plate_image, row - 1, col - 1)

        _, center_col, _ = st.columns([2, 1, 2])
        with center_col:
            if colony_img is not None and colony_img.size > 0:
                square_colony_img = make_square(colony_img, size=200)
                st.image(cv2.cvtColor(square_colony_img, cv2.COLOR_BGR2RGB), caption="Colony to Tag", use_container_width=True)
            else:
                st.warning("Could not extract colony image.")

        # Display example images as buttons
        st.write("Which example is most similar?")

        image_buttons = example_images
        prefixes = ["A.", "B.", "C.", "D."]

        cols = st.columns(len(image_buttons))
        for i, (name, img) in enumerate(image_buttons):
            with cols[i]:
                st.image(img, caption=name, width=150)
                button_caption = f"{prefixes[i]} {name}"
                if st.button(button_caption, key=f"btn_{name}_{idx}", use_container_width=True):
                    st.session_state.annotations[(row, col)] = button_caption
                    st.success(f"Tagged as '{name}'.")
                    if st.session_state.current_colony_idx < len(st.session_state.zero_colonies) - 1:
                        st.session_state.current_colony_idx += 1
                        st.rerun()
                    else:
                        st.info("All colonies have been tagged!")
                        st.rerun()

        if st.button("E. Not correct image", key=f"btn_not_correct_{idx}", use_container_width=True):
            st.session_state.annotations[(row, col)] = "E. Not correct image"
            st.success("Tagged as 'Not correct image'.")
            if st.session_state.current_colony_idx < len(st.session_state.zero_colonies) - 1:
                st.session_state.current_colony_idx += 1
                st.rerun()
            else:
                st.info("All colonies have been tagged!")
                st.rerun()

        st.write("---")
        # Navigation buttons — centred
        _, nav_prev, nav_next, _ = st.columns([2, 1, 1, 2])
        with nav_prev:
            if st.button("⬅️ Previous", use_container_width=True) and idx > 0:
                st.session_state.current_colony_idx -= 1
                st.rerun()
        with nav_next:
            if st.button("Next ➡️", use_container_width=True) and idx < len(st.session_state.zero_colonies) - 1:
                st.session_state.current_colony_idx += 1
                st.rerun()
        
        # --- Plate Navigation ---
        st.write("---")
        if st.button("Go to Next Plate ⏩", use_container_width=True):
            st.session_state.go_to_next_plate = True
            st.rerun()

        # Display progress and save option
        st.write("---")
        st.write(f"**Progress:** {len(st.session_state.annotations)} / {len(st.session_state.zero_colonies)} tagged.")
        
        if len(st.session_state.annotations) == len(st.session_state.zero_colonies):
            st.balloons()
            st.header("Tagging Complete!")
            
            selected_condition = st.session_state.get('selected_condition', 'condition')
            selected_plate = st.session_state.get('selected_plate', 'plate')
            selected_batch = st.session_state.get('selected_batch', 'batch')

            df_annotations = pd.DataFrame([
                {
                    'condition': selected_condition,
                    'plate': selected_plate,
                    'batch': selected_batch,
                    'row_1536': r, 'column_1536': c,
                    'row_384': (r + 1) // 2, 'column_384': (c + 1) // 2,
                    'tag': tag
                }
                for (r, c), tag in st.session_state.annotations.items()
            ])
            
            # Auto-save the file
            filename = f"{selected_condition}_{selected_plate}_{selected_batch}_annotations.csv"
            filepath = os.path.join(annotations_directory, filename)
            df_annotations.to_csv(filepath, index=False)
            st.success(f"Annotations automatically saved to `{filepath}`")

            st.session_state.plate_finished = True
            st.rerun()
    elif st.session_state.get('zero_colonies') == "COMPLETED":
        st.info("All colonies for the selected plate have been processed or no zero-size colonies were found.")
        if st.button("Go to Next Plate ⏩", use_container_width=True):
            st.session_state.go_to_next_plate = True
            st.rerun()

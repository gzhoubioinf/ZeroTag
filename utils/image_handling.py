import cv2

def crop_img(image_path):
    """ Loads an image from disk using OpenCV and returns it as a NumPy array in BGR order."""
    return cv2.imread(image_path)

def find_grid_by_cell_contours(img, num_rows=32, num_cols=48):
    """
    Directly detects the grid by finding the contours of individual cells and
    reconstructing the grid geometry from them. This is a robust method.

    Returns:
        - A tuple (grid_x, grid_y) for the top-left corner of the grid's content area.
        - A tuple (avg_cell_w, avg_cell_h) for the average size of a single cell.
    """
    # --- 1. Isolate Grid Structure ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use adaptive thresholding to handle lighting variations and highlight cell borders
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 2)

    # --- 2. Find All Potential Shapes ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # --- 3. Identify the "Cells" ---
    img_h, img_w = img.shape[:2]
    expected_cell_h, expected_cell_w = img_h / num_rows, img_w / num_cols
    valid_cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter based on size and aspect ratio to find cell-like shapes
        if (0.5 * expected_cell_w < w < 1.5 * expected_cell_w) and \
           (0.5 * expected_cell_h < h < 1.5 * expected_cell_h) and \
           (0.7 < w/h < 1.3):
            valid_cells.append((x, y, w, h))

    if len(valid_cells) < (num_rows * num_cols * 0.25): # Require at least 25% of cells to be found
        # Fallback if not enough cells are detected
        print("Warning: Contour detection failed to find enough cells. Falling back to simple division.")
        return (0, 0), (img_w // num_cols, img_h // num_rows)

    # --- 4. Reconstruct the Grid ---
    # Get the top-left and bottom-right corners of the entire grid area
    all_x = [c[0] for c in valid_cells]
    all_y = [c[1] for c in valid_cells]
    grid_x = min(all_x)
    grid_y = min(all_y)
    grid_w = max([c[0] + c[2] for c in valid_cells]) - grid_x
    grid_h = max([c[1] + c[3] for c in valid_cells]) - grid_y

    # Calculate the average cell size from the reconstructed grid
    avg_cell_w = int(grid_w / num_cols)
    avg_cell_h = int(grid_h / num_rows)

    return (grid_x, grid_y), (avg_cell_w, avg_cell_h)


def extract_colony(img, row, col, num_rows=32, num_cols=48):
    """
    Extracts a clean image of a single grid cell's contents, without attempting to recenter.
    This provides a reliable view of the specified location.
    """
    img_height, img_width = img.shape[:2]
    
    # 1. Use the robust direct detection method to get grid parameters
    grid_origin, cell_size = find_grid_by_cell_contours(img, num_rows, num_cols)
    grid_offset_x, grid_offset_y = grid_origin
    fixed_cell_width, fixed_cell_height = cell_size

    # Define a small inset to avoid capturing the grid lines themselves
    inset = 2  # pixels

    # 2. Calculate the top-left corner of the target cell's content area
    x_start = grid_offset_x + (col * fixed_cell_width) + inset
    y_start = grid_offset_y + (row * fixed_cell_height) + inset

    # 3. Define the crop window, reduced by the inset on both sides
    crop_w = fixed_cell_width - (2 * inset)
    crop_h = fixed_cell_height - (2 * inset)

    # Ensure crop window is valid and within image bounds
    x_start = max(0, min(x_start, img_width - crop_w))
    y_start = max(0, min(y_start, img_height - crop_h))
    
    # Fallback if inset is too large for the cell
    if crop_w <= 0 or crop_h <= 0:
        x_start = grid_offset_x + (col * fixed_cell_width)
        y_start = grid_offset_y + (row * fixed_cell_height)
        crop_w = fixed_cell_width
        crop_h = fixed_cell_height

    # Cast to integers for slicing
    x_start, y_start = int(x_start), int(y_start)
    crop_h, crop_w = int(crop_h), int(crop_w)

    # Extract the clean cell image and return it directly
    cell = img[y_start : y_start + crop_h, x_start : x_start + crop_w]

    return cell


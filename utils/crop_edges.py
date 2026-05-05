import cv2
import numpy as np
import os
import glob

# A simple progress bar without needing a new library
def print_progress(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total: 
        print()

def detect_grid_bounds(img, padding_percent=0.01):
    """
    Automatically detects the main grid of colonies in an image, ignoring borders.
    Returns the (x, y, width, height) of the detected grid.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None # Return None if no grid is found
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add a small amount of padding to prevent clipping the edges
    padding_x = int(w * padding_percent)
    padding_y = int(h * padding_percent)

    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    w = min(img.shape[1] - x, w + (2 * padding_x))
    h = min(img.shape[0] - y, h + (2 * padding_y))

    return x, y, w, h

def main():
    """
    Main function to process all images in the specified directory.
    """
    image_directory = '../../data/plate_images'
    image_pattern = os.path.join(image_directory, '*.JPG.grid.jpg')
    image_files = glob.glob(image_pattern)
    total_files = len(image_files)

    if not image_files:
        print(f"No images found in '{image_directory}'. Please check the path.")
        return

    print(f"Found {total_files} images to process.")

    for i, filepath in enumerate(image_files):
        print_progress(i + 1, total_files, prefix='Progress:', suffix='Complete', length=50)
        
        img = cv2.imread(filepath)
        if img is None:
            print(f"\nWarning: Could not read image {filepath}. Skipping.")
            continue

        bounds = detect_grid_bounds(img)
        
        if bounds is None:
            print(f"\nWarning: No grid detected in {filepath}. Skipping.")
            continue
            
        x, y, w, h = bounds
        cropped_img = img[y:y+h, x:x+w]
        
        # Overwrite the original file with the cropped version
        cv2.imwrite(filepath, cropped_img)
        
    print("\nProcessing complete. All images have been cropped.")

if __name__ == '__main__':
    main()

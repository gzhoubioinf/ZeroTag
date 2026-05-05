# ZeroTag

A Streamlit app for tagging zero-size colonies in high-throughput bacterial growth plate images. Each plate image is paired with an IRIS measurement file containing colony size data. Colonies measured as size 0 are displayed one by one for manual classification.

---

## Requirements

- Python 3.11 (required — other versions not compatible)
- Miniconda or Anaconda

---

## Installation

```bash
# Create and activate a conda environment with Python 3.11
conda create -n zerotag python=3.11
conda activate zerotag

# Navigate to the project folder
cd /path/to/ZeroTag

# Install dependencies
pip install setuptools
pip install -r requirements.txt
```

---

## Running the App

```bash
conda activate zerotag
cd /path/to/ZeroTag
streamlit run app/main.py
```

The app opens automatically in your browser.

---

## How to Use

### 1. Select your person
In the left sidebar, choose **Person 1** or **Person 2**. This loads your assigned set of plates.

| | Iris files | Zero-size colonies to tag |
|---|---|---|
| Person 1 | 463 | 34,209 |
| Person 2 | 445 | 34,209 |

Your file list is in `data/person1_filelist.txt` or `data/person2_filelist.txt`.

### 2. Select a plate
Use the sidebar dropdowns to choose:
- **Condition** — the growth condition (e.g. `Ampicillin-128ugml`)
- **Plate Number** — replicate number (1–4)
- **Batch** — imaging batch

### 3. Tag colonies
The app automatically finds all zero-size colonies on the selected plate and presents them one by one. For each colony you see:
- A cropped image of the colony
- The full plate image with the colony highlighted (expandable)
- Four reference examples to compare against

Click the button that best matches the colony:

| Option | Meaning |
|---|---|
| A. No Growth | Well is empty — no colony visible |
| B. Poor Growth | Very small or faint colony |
| C. Normal Growth | Colony grew normally |
| D. Over Growth | Colony is abnormally large |
| E. Not correct image | Image extraction failed or unclear |

### 4. Navigation
- **Previous / Next** — move between colonies on the current plate
- **Go to Previous Plate / Go to Next Plate** — jump between plates
- **Reprocess Plate** — reload the current plate from scratch

### 5. Saving
Annotations are **saved automatically** when all zero-size colonies on a plate are tagged. Files are saved to:

```
annotations/{condition}_{plate}_{batch}_annotations.csv
```

Each CSV contains: `condition`, `plate`, `batch`, `row_1536`, `column_1536`, `row_384`, `column_384`, `tag`.

---

## Project Structure

```
ZeroTag/
├── app/
│   ├── main.py              # App entry point
│   ├── example_tagging.py   # Zero-size colony tagging UI
│   ├── colony_picker.py     # Colony picker UI
│   └── utils.py             # File listing utilities
├── config/
│   ├── config_person1.yaml  # Config for Person 1
│   └── config_person2.yaml  # Config for Person 2
├── data/
│   ├── person1/
│   │   ├── iris_measurements/   # IRIS files (463)
│   │   └── plate_images/        # Plate images (463)
│   ├── person2/
│   │   ├── iris_measurements/   # IRIS files (445)
│   │   └── plate_images/        # Plate images (445)
│   ├── Example_images/          # Reference colony images
│   ├── person1_filelist.txt
│   └── person2_filelist.txt
├── annotations/             # Auto-saved annotation CSVs
├── utils/
│   ├── data_loading.py
│   └── image_handling.py
└── requirements.txt
```

---

## Data Format

**IRIS files** (`.iris`): tab-separated, one row per colony (1536 colonies per plate, 32×48 grid). Key column: `colony size` — a value of `0` indicates the colony was not detected.

**Plate images** (`.JPG.grid.jpg`): processed grid images paired 1:1 with IRIS files by filename stem.

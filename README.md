# Computer Vision — Assignments Repository

A collection of hands-on computer vision assignments implemented in Python using
[OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), and
[Matplotlib](https://matplotlib.org/). Each assignment is a self-contained Jupyter
notebook that demonstrates a fundamental image-processing concept, from geometric
transformations to spatial-domain filtering.

## Table of Contents

- [Project Description](#project-description)
- [Assignments Overview](#assignments-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

## Project Description

This repository is a practical learning path through core computer-vision topics.
Every assignment pairs concise theory with runnable code so you can see how
mathematical concepts (transformation matrices, color-space mappings, convolution
kernels) translate into working OpenCV calls.

The assignments cover:

1. **Geometric Transformations** — translation, rotation, and scaling via affine
   matrices (`cv2.warpAffine`, `cv2.getRotationMatrix2D`, `cv2.resize`).
2. **Color Space Conversions** — moving between BGR, RGB, Grayscale, and HSV, and
   analyzing the individual HSV channels.
3. **Spatial Domain Filtering** — smoothing (Box and Gaussian filters) and
   sharpening (Laplacian filter) using convolution kernels.

All notebooks operate on a shared sample image (`BMW.jpeg`) and render their
results with Matplotlib.

## Assignments Overview

| Folder | Notebook | Topic | Key OpenCV Functions |
|--------|----------|-------|----------------------|
| `CV_Assignment_2/` | `Ass2.ipynb` | Geometric Transformations (Translation, Rotation, Scaling) | `cv2.warpAffine`, `cv2.getRotationMatrix2D`, `cv2.resize` |
| `CV_Assignment_3/` | `Ass3.ipynb` | Color Space Conversions (BGR/RGB/Gray/HSV) | `cv2.cvtColor`, `cv2.split` |
| `CV_Assignment_4/` | `assignment4.ipynb` | Spatial Domain Filtering (Box, Gaussian, Laplacian) | `cv2.blur`, `cv2.GaussianBlur`, `cv2.Laplacian`, `cv2.addWeighted` |

Each assignment folder also contains its own `README.md` with detailed theory,
per-cell code explanation, and a Q&A section.

## Prerequisites

- **Python 3.8+** (tested with CPython 3.x)
- **pip** (Python package manager)
- A Jupyter environment to run the notebooks:
  - [Jupyter Notebook](https://jupyter.org/install),
  - [JupyterLab](https://jupyterlab.readthedocs.io/), or
  - the [VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
    Python extension.

The only third-party dependencies are:

| Package | Purpose |
|---------|---------|
| `opencv-python` | Computer-vision primitives (`cv2`); loads and transforms images |
| `numpy` | Array/matrix operations for transformation kernels |
| `matplotlib` | Rendering results in subplot grids |

> The `cv2` module is provided by the `opencv-python` PyPI package.

## Installation

1. **Clone the repository** (or download and extract it):

   ```bash
   git clone <repository-url>
   cd "Computer-Vision-"
   ```

2. **(Recommended) Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install the dependencies:**

   Using the provided requirements file:

   ```bash
   pip install -r requirement.txt
   ```

   Or install them manually:

   ```bash
   pip install opencv-python numpy matplotlib
   ```

4. **Install Jupyter** (if not already available):

   ```bash
   pip install notebook
   ```

## Usage

Each assignment is a standalone notebook. To run one:

1. Launch Jupyter from the repository root:

   ```bash
   jupyter notebook
   ```

   This opens a browser window listing the assignment folders.

2. Open an assignment notebook, for example
   `CV_Assignment_2/Ass2.ipynb`, and run the cells **in order**
   (menu: *Run* → *Run All*, or `Shift+Enter` per cell).

3. The notebook displays its output (e.g. a 2×2 grid of transformed images)
   inline.

### Example — running Assignment 2 headlessly

If you prefer to execute a notebook from the command line (e.g. for CI or
automation), use `nbconvert`:

```bash
pip install nbconvert
jupyter nbconvert --to notebook --execute CV_Assignment_2/Ass2.ipynb --output Ass2_executed.ipynb
```

### Using your own image

All notebooks read a file named `BMW.jpeg` from their own folder. To use a
different image, either:

- Place your image in the assignment folder and rename it to `BMW.jpeg`, or
- Edit the `cv2.imread('BMW.jpeg')` call (or the `process_image('BMW.jpeg')`
  argument in Assignment 3) to point at your file.

## Project Structure

```
Computer-Vision-/
├── requirement.txt              # Python dependencies (opencv-python, numpy, matplotlib)
├── README.md                    # This file
├── CV_Assignment_2/
│   ├── Ass2.ipynb               # Geometric transformations notebook
│   ├── BMW.jpeg                 # Sample input image
│   ├── output.png               # Cached output visualization
│   └── README.md                # Assignment 2 documentation
├── CV_Assignment_3/
│   ├── Ass3.ipynb               # Color space conversions notebook
│   ├── BMW.jpeg                 # Sample input image
│   ├── output.png               # Cached output visualization
│   └── README.md                # Assignment 3 documentation
└── CV_Assignment_4/
    ├── assignment4.ipynb        # Spatial domain filtering notebook
    ├── BMW.jpeg                 # Sample input image
    ├── Box_Filtered.png         # Example box-filtered output
    ├── Gaussian_Filtered.png    # Example Gaussian-filtered output
    ├── Kernel.png               # Example kernel visualization
    ├── Sharpened_Image.png      # Example sharpened output
    └── README.md                # Assignment 4 documentation
```

## Configuration

There is no external configuration file. All parameters (translation offsets,
rotation angles, scaling factors, kernel sizes, sigma values) are defined as
plain Python variables at the top of the relevant notebook cells. To experiment:

- **Assignment 2:** change `tx, ty`, `angle`, `sx, sy` before running the
  transformation cells.
- **Assignment 3:** pass a different image path to `process_image(...)`.
- **Assignment 4:** change the kernel size (e.g. `(5, 5)` → `(9, 9)`) or the
  `addWeighted` weights to control blur/sharpen strength.

## Running Tests

This repository does not ship a formal automated test suite. Each notebook is
self-validating: running all cells produces the expected visualizations, and the
notebooks guard against missing files (e.g. checking `if image is None`).

If you want to add automated checks, a lightweight approach is to execute every
notebook headlessly and fail on errors:

```bash
pip install nbconvert nbclient
for nb in CV_Assignment_2/Ass2.ipynb CV_Assignment_3/Ass3.ipynb CV_Assignment_4/assignment4.ipynb; do
  jupyter nbconvert --to notebook --execute --stdout "$nb" > /dev/null
done
```

## Contributing

Contributions are welcome. To add or improve an assignment:

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b assignment-5-my-topic
   ```
2. Add a new folder `CV_Assignment_N/` with:
   - a Jupyter notebook implementing the concept,
   - a `README.md` documenting the theory and code,
   - any required sample assets (e.g. a sample image).
3. Keep dependencies limited to the packages in `requirement.txt` where possible.
4. Run the notebook end-to-end to confirm outputs render correctly.
5. Update this root `README.md` (the [Assignments Overview](#assignments-overview)
   table and [Project Structure](#project-structure)) to reflect your addition.
6. Open a pull request describing the change and its learning objectives.

Please follow the existing code style: import `cv2`, `numpy as np`, and
`matplotlib.pyplot as plt`; convert BGR→RGB before displaying with Matplotlib;
and include a `None` check after `cv2.imread`.

## License

This repository is provided for educational purposes. See individual assignment
folders for any per-assignment notes.

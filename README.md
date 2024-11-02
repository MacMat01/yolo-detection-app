# Cards Object Detection

[![CodeQL](https://github.com/MacMat01/cards-object-detection/actions/workflows/codeql.yml/badge.svg)](https://github.com/MacMat01/cards-object-detection/actions/workflows/codeql.yml)

This project aims to detect strategic fruits cards using YOLOv8. It is implemented in Python and uses several libraries
for data processing and model training.

## Project Structure

The project has the following structure:
- `card_detection_app.py`: Main application script for card detection.
- `environment.yml`: Contains the conda environment configuration.
- `setup.py`: Contains the setup configuration for the Python package.

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:

- [Python](https://www.python.org/downloads/) 3.12.3
- [Conda package manager](https://www.anaconda.com/download)
- [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) 12.3
- Suggested IDE:
  - [VSCode](https://code.visualstudio.com/Download) with Python and Jupyter extensions
  - [PyCharm Professional](https://www.jetbrains.com/pycharm/download/?section=windows) is also a good choice if you
    have a license

### Installation

1. Clone the repository:

```bash
git clone https://github.com/MacMat01/yolo-detection-app.git
```

2. Navigate to the project directory:

```bash
cd yolo-detection-app
```

3. Create a new conda environment from the `environment.yml` file:

```bash
conda env create --name <your-environment-name>
```

4. Activate the conda environment:

```bash
conda activate <your-environment-name>
```

5. Install the `build` and `pip` tools:

```bash
pip install --upgrade build pip
```

6. Build a source distribution (sdist) and a binary distribution (wheel) of your package:

```bash
python -m build
```

7. Install the package from the wheel file:

```bash
pip install --user dist/*.whl # If it doesn't work, change </*.whl> to the name of the wheel file generated in step 6
```

### Running the Application

#### For Using the Main Application

To run the main application, navigate to the `src` directory and run the `card_detection_app.py` script:

```bash
cd src
```

Before running the main application, in the CardDetectionApp’s init method,
remember to change the `video_path` to the path of the video you want to use.
Otherwise, by default will be used the video camera

For the main application, run the following command:
```bash
python card_detection_app.py
```

## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

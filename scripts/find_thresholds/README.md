# find_thresholds.py

## Description
The `find_thresholds.py` script is designed to derive linear regression models for the BRISK, KAZE, and AKAZE feature detectors. These models aim to establish the correspondence between the average number of detected keypoints in an image and the threshold value applied by the detectors. The script employs a diverse image dataset with optional axis-aligned bounding boxes representing regions where features should not be detected. Additionally, histogram normalization using CLAHE may influence the models, leading to the creation of separate models for all combinations of mask usage and CLAHE application. The resulting threshold models play a crucial role in configuring the Stabilo module, enabling a fair comparison with other detectors like ORB, SIFT, and RSIFT.

## Usage
```
python find_thresholds.py --dataset-dir=<dir>

Options
--dataset-dir=<dir>: Directory containing the diverse image dataset (default: scenes/).
```

## Dataset Requirements

1. **Diverse Set of Images**: The dataset should include a diverse set of images representing various scenarios.
2. **Axis-Aligned Bounding Boxes (Optional)**: The images may come with axis-aligned bounding boxes representing regions where features should not be detected.

## Script Functionality
- **Threshold Models**: Finds linear models for BRISK, KAZE, and AKAZE detectors, considering various combinations of detectors, mask usage, and CLAHE.
- **Data Generation**: Collects data by finding the average number of detected keypoints for different threshold values.
- **Model Fitting**: Fits linear models to the collected data, considering a specified range of keypoints.
- **Data Filtering**: Filters the collected data to fit the model only within a specified range of keypoints.
- **Data and Model Storage**: Saves raw and filtered data, as well as the linear models for further analysis.
- **Plotting**: Generates and saves plots illustrating the relationship between average keypoints and thresholds.

## Important Notes
- The dataset should be carefully crafted to ensure a representative set of images with diverse content and scenarios.
- Axis-aligned bounding boxes can be provided to guide the feature detection process and create more meaningful threshold models.
- The generated models will be stored in the `models` directory, while the data and plots will be saved in the `results` and `plots` directories, respectively.

Feel free to adapt the script and instructions based on your specific dataset and requirements.

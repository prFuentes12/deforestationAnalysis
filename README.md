# Deforestation Analysis through Satellite Images

## Project Overview

This project aims to analyze deforestation in the Amazon rainforest, specifically in the Jamanxim National Forest in Brazil, using satellite images extracted from a GIF. The analysis spans from 2000 to 2019, offering a temporal study of deforestation in the region. Various image processing techniques were applied to assess deforestation, such as intensity adjustment, adaptive thresholding, and morphological operations for segmentation. The deforested areas were quantified in square kilometers, providing a detailed view of the forest's degradation over time.

The core of the analysis involves:
1. **Extracting Frames from a GIF**: The GIF, representing images from the years 2000 to 2019, is extracted, and each frame corresponds to one year of satellite imagery.
2. **Image Preprocessing**: Images are converted to grayscale, a median filter is applied to reduce noise, and areas affected by clouds are removed to improve segmentation.
3. **Deforestation Segmentation**: The deforested areas are isolated using adaptive thresholding, followed by morphological cleaning to refine the segmented image.
4. **Deforested Area Calculation**: The area affected by deforestation is computed in square kilometers by counting the deforested pixels.

The results include visualizations that show the progression of deforestation and a graph that tracks the changes in forest cover over the years.

## Tools and Techniques Used

### Programming Environment
- **Python 3.10.12**: The project was developed using this Python version in Google Colab.
- **IDE**: Google Colab for real-time code execution and visualization.

### Libraries
- **OpenCV (cv2)**: Used for image loading, grayscale conversion, median filtering, segmentation via adaptive thresholding, and morphological operations.
- **NumPy**: Employed for pixel-based operations such as counting deforested pixels and calculating area in square kilometers.
- **Matplotlib**: For visualizing the results with graphs.
- **Pillow (PIL)**: For handling images in various formats.
- **python-docx**: Used to manipulate Word documents for extracting embedded GIF images.

## Project Steps

### 1. Extracting GIF Frames
The GIF representing satellite images from 2000 to 2019 is extracted from a Word document. Each frame represents an image from a different year, showing the state of the forest.

### 2. Image Preprocessing
Several preprocessing techniques were used to improve image quality:
- **Grayscale Conversion**: Simplifies the image and focuses on intensity values.
- **Median Filter**: Reduces noise, particularly in low-light areas.
- **Cloud Removal**: The cloud-covered areas were detected using intensity thresholds and then inpainted to eliminate their effect on the analysis.

### 3. Deforestation Segmentation
- **Adaptive Thresholding**: This technique adjusts the threshold for each pixel based on local conditions, allowing for effective segmentation of deforested areas.
- **Morphological Opening**: Used to clean the segmentation results by removing small noise particles and improving the definition of deforested regions.

### 4. Calculating Deforested Area
The deforested area in square kilometers is calculated by counting the number of pixels in the segmented regions and applying a pixel-to-kilometer ratio.

### 5. Visualizing Results
The processed images, including original, preprocessed, segmented, and deforested area calculations, are displayed. A graph is generated to show the evolution of deforestation over the years.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/deforestation-pipeline.git

2. Run the pipeline:
   python deforestation.py

## Data Used
The project uses a set of satellite images obtained from NASA’s Earth Observatory. A total of 20 images from the Jamanxim National Forest were analyzed, spanning from 2000 to 2019. The images exhibit significant variability in lighting conditions and forest cover, presenting challenges for accurately identifying deforested areas.

## Key Techniques

- **Image Preprocessing**:
  - Grayscale conversion simplifies the analysis.
  - Median filter helps reduce noise.
  
- **Cloud Detection and Removal**: 
  - Inpainting methods to remove cloud-affected areas.

- **Adaptive Thresholding**: 
  - Allows segmentation under varying lighting conditions.

- **Morphological Operations (Opening)**: 
  - Used to refine the segmentation.

- **Deforested Area Calculation**: 
  - The deforested area is computed by counting deforested pixels and converting them to square kilometers.

## Results
The analysis shows a clear trend of increasing deforestation in the Jamanxim National Forest from 2000 to 2019. The segmented areas, based on thresholding and morphological operations, reveal the regions most affected by deforestation. The graph of deforested area over time highlights a steady increase in forest loss.

However, some segmentation errors occurred due to atmospheric conditions, such as cloud cover, which can interfere with accurate segmentation. To improve future analyses, further research into advanced cloud detection techniques or machine learning models could be considered.

## Future Work
- Investigate more advanced methods for cloud detection or use machine learning models to distinguish between clouds and vegetation.
- Use higher resolution satellite images to reduce errors caused by atmospheric conditions.
- Explore real-time deforestation monitoring techniques using similar methods.

## Authors
- Fernando Jesçus Fuentes Carrasco

## References
NASA Earth Observatory. (n.d.). Making sense of Amazon deforestation patterns. [Link to article]

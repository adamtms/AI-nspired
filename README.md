![Computer Vision](https://img.shields.io/badge/Computer%20Vision-%E2%9C%94-brightgreen)
![ResNet50](https://img.shields.io/badge/ResNet50-%E2%9C%94-blue)
![DINOv2](https://img.shields.io/badge/DINOv2-%E2%9C%94-purple)
![Statistical Analysis](https://img.shields.io/badge/Statistical%20Analysis-%E2%9C%94-orange)

# AI'nspired

Software for analysis of images for elements of inspiration coinciding with elements of the final designs.

## Table of content

- [Authors](#authors)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Content of the Repository](#content-of-the-repository)
- [How to Use](#how-to-use)

## Authors

- [Adam Tomys](https://github.com/adamtms)
- [Marcel Rojewski](https://github.com/marcelrojo)
- [Marcin Kapiszewski](https://github.com/Marcin59)
- [Jakub Jagła](https://github.com/j-millet)
- [Łukasz Borak](https://github.com/B0cz3k)

## Project Overview

This project investigates the impact of different sources of visual inspiration on creative outcomes, comparing AI-generated images (Midjourney V6) with web-sourced images. The similarity between inspiration images and final student design projects was quantified using computer vision techniques and deep learning embeddings.

## Dataset

The experiments were conducted on data collected from projects developed for the "Trams of the Future" competition. The dataset comprises 26 tram designs, each accompanied by up to 10 AI-generated inspirations and 10 web-sourced inspirations selected by participants during the design process. The resulting dataset consists of:
- 474 web inspirations
- 417 AI inspirations
- 159 final project images

## Methodology

To quantify the influence of inspirations, computational similarity scores were calculated between the final project images and the gathered inspirations. The analysis employed both classical image comparison methods (color and contrast similarity) and deep learning embeddings using **ResNet50** and **DINOv2** models. The computational results were then compared with participants' declared inspirations and evaluated against expert rankings.

## Results

For each project, the top-ranked inspiration from both sources was selected, and the similarity scores were statistically compared using the Mann-Whitney U test.  The statistical significance of the observed differences was evaluated at α = 0.05.

![Plot Results](https://github.com/user-attachments/assets/a4a2ca1f-1215-4a7f-b445-3f29cb821f88)

| Metric               | p-value   | Interpretation                              |
|---------------------|-----------|---------------------------------------------|
| Color Similarity    | 3.99e-20 | Web inspirations show significantly higher color similarity |
| Contrast Similarity | 0.475     | No significant difference                   |
| ResNet Embedding    | 4.68e-04 | Web inspirations show significantly higher semantic similarity (ResNet) |
| DINO Embedding      | 9.37e-04 | AI inspirations show significantly higher semantic similarity (DINO) |

## Content of the Repository

- `app` directory: Contains the Streamlit app code for visualizing computed similarity scores for each group.
- `modules` directory: Contains subdirectories corresponding to different similarity metrics: `color`, `contrast`, `dino`, and `resnet`. Each subdirectory implements a ComparisonModule class that calculates image similarity according to the respective metric.
- `scripts` directory: Includes the `dataset_transformer.py` script for restructuring the image dataset into the required directory format
- `notebooks` directory: Contains various Jupyter notebooks used for testing different approaches, analyzing results, and exploring methods later applied in the final pipeline.
- `csv_generation_script.py`: Script responsible for computing similarity scores between each final submission image and its suggested inspirations across all measured metrics, storing the results in a CSV file.

## How to Use

1. Store your dataset in the data directory, following the format outlined in `scripts/dataset_transformer.py`. If needed, run the script to automatically restructure your current file hierarchy into the expected format
2. Execute `csv_generation_script.py` to calculate similarity scores and generate the CSV file containing the computed values
3. Visualize and analyze the computed similarity scores using `notebooks/group_scores.ipynb`  

Remarks:
- To run the Streamlit app, ensure that all image files are downloaded and placed inside the static directory.
- Before running any notebooks, set Jupyter: Notebook File Root to `${workspaceFolder}` to ensure proper file path resolution.
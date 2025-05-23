<div align="center">

# Digital Earth for a Resilient Caribbean 
Mapping Housing Stock Characteristics from VHR Aerial Images for Climate Resilience in the Caribbean

<p>
<b><a href="#-description">Description</a></b>
|
<b><a href="#-dataset">Dataset</a></b>
|
<b><a href="#-code-organization">Code Organization</a></b>
|
<b><a href="#-usage">Usage</a></b>
|
<b><a href="#-file-organization">File Organization</a></b>
|
<b><a href="#acknowledgement">Acknowledgment</a></b>
|
<b><a href="#citation">Citation</a></b>
</p>

</div>

## 📜 Description
This work leverages computer vision techniques for the automated classification of rooftop  characteristics from very very high-resolution (VHR) drone images, RGB orthophotos, and airborne LiDAR data. 

The Caribbean is one of the most vulnerable regions to climate change. In 2017, Category 5 Hurricane Maria  destroyed ~90% of Dominica’s housing stock, accumulating damages > 380M USD in the housing sector alone. The recent disasters have spurred ambitious climate resilience initiatives such as the Resilient Housing Scheme by the Government of the Commonwealth of Dominica strives to make 90% of housing stock resilient by 2030. 

The Digital Earth for a Resilient Caribbean, a [World Bank](https://www.worldbank.org/en/home) Project funded by [GFDRR](https://www.gfdrr.org/en), aims help governments produce more timely housing stock information to improve resilience and disaster response in the Caribbean. The overarching goal of the project is to enhance local capacity to leverage EO-based solutions in support of resilient infrastructure and housing operations. 
<p>
<img src="./assets/results.png" width="70%" height="70%" />

This code accompanies the following paper(s):
- Tingzon, I., Cowan, N. M., & Chrzanowski, P. (2023). Fusing VHR Post-disaster Aerial Imagery and LiDAR Data for Roof Classification in the Caribbean. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (pp. 3740-3747).
- Tingzon, I., Cowan, N. M., & Chrzanowski, P. (2023). Mapping Housing Stock Characteristics from Drone Images for Climate Resilience in the Caribbean. Tackling Climate Change with Machine Learning Workshop at NeurIPS 2023. arXiv preprint arXiv:2312.10306.

## 📂 Dataset
To generate our ground truth dataset, we used the following three data sources for Dominica and Saint Lucia: (1) VHR aerial imagery, (2) LiDAR data (optional), and (3) building footprints in the form of georeferenced vector polygons.

We annotated ~15k buildings according to two attributes: (1) roof type and (2) roof material. The following figures below illustrate examples of the RGB orthophoto and LiDAR-derived image patches.
<p>
<img src="./assets/roof-characteristics.png" width="70%" height="70%" />

For data protection purposes, we did not include in this repo the labelled training data. To access the dataset, please reach out at tisabelle@worldbank.org. To use your own data, kindly follow the recommended <a href="#-file-organization">file organization</a>.

## 💻 Getting Started
See the following Colab notebooks to get started:
- Part 1: [Building Footprint Delineation for Disaster Risk Reduction and Response (DRR)](https://colab.research.google.com/github/GFDRR/caribbean-rooftop-classification/blob/master/tutorials/01_building_delineation.ipynb)
- Part 2: [Rooftop Type and Roof Material Classification using Drone Imagery](https://colab.research.google.com/github/GFDRR/caribbean-rooftop-classification/blob/master/tutorials/02_building_classification.ipynb)

## 💻 Code Organization 

This repository is divided into the following main folders and files:
- **notebooks/**: contains all Jupyter notebooks for exploratory data analysis and model prediction.
- **tutorials/**: contains runnable Google Colab notebooks for (1) building footprint delineation using SAM and (2) rooftop classification using CNNs.
- **utils/**: contains utility methods for loading datasets, building model, and performing training routines. 
- **data.py**: script for generating labelled image tiles based on the parameters specified in the yaml file located in `configs/data/`
- **train.py**: script for training the CNN model based on the parameters specified in the yaml file located in `configs/cnn/` (supports ResNet50, EfficientNet-B0, InceptionV3, and VGG-16)
- **fusion.py**: script for training the downstream ML classifier that combines the CNN models trained on RGB and LiDAR data based on the parameters specified in the yaml file located in `configs/fusion/` (supports logistic regression, random forest, and linear SVC)


## 💻 Usage

### Setup
Clone the repository and then create and activate a new conda environment:
```s
conda create -n envname
conda activate envname
pip install -r requirements.txt
```

### Data Preparation
To generate the dataset, run:
```s
python data.py \
--config="config/data/<DATA_CONFIG>.yaml"
```

### CNN Model Training
To train the CNN model, run:
```s
python train.py \
--exp_config="config/cnn/<CNN_CONFIG>.yaml"
```

### Data Fusion (RGB + LiDAR)
To train the data fusion model (RGB + LiDAR), run:
```s
python fusion.py \
--exp_config="config/fusion/<FUSION_CONFIG>.yaml"
```

## 📂 File Organization 
The datasets are organized as follows:
```
data
├── csv
│   ├── RGB_DOM.csv
│   └── LIDAR_DOM.csv
├── rasters
│   ├── drone
│   │   ├── drone_colihaut_DOM.tif
│   │   └── ...
│   ├── lidar
│   │   ├── dsm_DOM.tif
│   │   ├── dtm_DOM.tif
│   │   └── ndsm_DOM.tif
│   ├── ortho
│   │   └── ortho_DOM.tif
│   └── tiles
│       ├── RGB_DOM
│       │   ├── DOM-975-NO_ROOF-INCOMPLETE.tif
│       │   └── ...
│       └── LIDAR_DOM
│           ├── DOM-975-NO_ROOF-INCOMPLETE.tif
│           └── ...
└── vectors
    ├── bldgs_ortho_DOM.gpkg
    └── ...
```

## Acknowledgement
This project builds on the work of the [Global Program for Resilient Housing](https://www.worldbank.org/en/topic/disasterriskmanagement/brief/global-program-for-resilient-housing) by the World Bank. We thank Mike Fedak, Chris Williams, and Sarah Antos for their assistance in providing access to the datasets as well as the insightful discussions on the data landscape in the Caribbean.


## Citation
If you find this repository useful, please consider giving a star ⭐ and citation 🦖:
```
@article{tingzon2023mapping,
  title={Mapping Housing Stock Characteristics from Drone Images for Climate Resilience in the Caribbean},
  author={Tingzon, Isabelle and Cowan, Nuala Margaret and Chrzanowski, Pierre},
  journal={arXiv preprint arXiv:2312.10306},
  year={2023}
}
```
```
@inproceedings{tingzon2023fusing,
  title={Fusing VHR Post-disaster Aerial Imagery and LiDAR Data for Roof Classification in the Caribbean},
  author={Tingzon, Isabelle and Cowan, Nuala Margaret and Chrzanowski, Pierre},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3740--3747},
  year={2023}
}
```

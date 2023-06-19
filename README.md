<div align="center">

# Digital Earth for a Resilient Caribbean 
Data Fusion of Post-disaster Aerial Imagery and LiDAR Data for Rooftop Classification using CNNs

</div>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

## 📜 Description
This work leverages convolutional neural networks (CNNs) and data fusion techniques for the automated classification of rooftop  characteristics from very high-resolution orthophotos and airborne LiDAR data. 

The Digital Earth for a Resilient Caribbean project aims help governments produce more timely building information to improve resilience and disaster response in the Caribbean.

## ⚙️ Local Setup for Development
This repo assumes the use of [conda](https://docs.conda.io/en/latest/miniconda.html) for simplicity in installing GDAL. More info on how to install GDAL can be found


## 🗄 File Organization
### Data Directory
The datasets are organized as follows:
```
data
├── csv
│   ├── roof_material.csv
│   └── roof_type.csv
├── rasters
│   ├── drone
│   │   └── drone_colihaut_DOM.tif
│   ├── lidar
│   │   ├── dsm_DOM.tif
│   │   ├── dtm_DOM.tif
│   │   └── ndsm_DOM.tif
│   ├── ortho
│   │   └── ortho_DOM.tif
│   └── tiles
│     	├── ndsm
│       │   ├── roof_material
│       │   │   ├── BLUE_TARP
│       │   │   ├── CONCRETE_CEMENT
│       │   │   ├── HEALTHY_METAL
│       │   │   ├── IRREGULAR_METAL
│       │   │   └── INCOMPLETE
│       │   └── roof_type
│       │       ├── FLAT
│       │       ├── GABLE
│       │       ├── HIP
│       │       └── NO_ROOF
│       └── ortho
│           ├── roof_material
│           └── roof_type
└── vectors
    ├── annotation_tiles_DOM.gpkg
    ├── building_footprints_DOM.gpkg
    ├── building_footprints_annotated_DOM.gpkg
    └── geoboundaries_DOM.gpkg
```
<div align="center">

# Digital Earth for a Resilient Caribbean 
Mapping Housing Stock Characteristics from VHR Aerial Images for Climate Resilience in the Caribbean

<p>
<b><a href="#-description">Description</a></b>
|
<b><a href="#-datasets">Dataset</a></b>
|
<b><a href="#-code-organization">Code Organization</a></b>
|
<b><a href="#-setup">Setup</a></b>
|
<b><a href="#-file-organization">File Organization</a></b>
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

## 📂 Datasets
To generate our ground truth dataset, we used the following three data sources for Dominica and Saint Lucia: (1) VHR aerial imagery, (2) LiDAR data (optional), and (3) building footprints in the form of georeferenced vector polygons.

We annotated ~15k buildings according to two attributes: (1) roof type and (2) roof material. The following figures below illustrate examples of the RGB orthophoto and LiDAR-derived image patches.
<p>
<img src="./assets/roof-characteristics.png" width="70%" height="70%" />

## Code Organization 

This repository is divided into three main parts:
- **notebooks/**: contains all Jupyter notebooks for exploratory data analysis.
- **tutorials/**: contains runnable Google Colab notebooks for (1) building footprint delineation using SAM and (2) rooftop classification using CNNs.
- **utils/**: contains utility methods for loading datasets, building model, and performing training routines. 

## 💻 Setup
To generate the dataset, run:
```s
python run.py --config="config/data/RGB_DOM.yaml"
```
You can replace the config by any of the yaml files in `configs/data/`.

To train the CNN model, run:
```s
python train.py --exp_config="config/cnn/cnn-roof_material-efficientnetb0-LIDAR_DOM.yaml"
```
You can replace the config by any of the yaml files in `configs/cnn/`.

To train the data fusion model, run:
```s
python fusion.py --exp_config="config/fusion/fusion_LR_embeds.yaml"
```
You can replace the config by any of the yaml files in `configs/fusion/`.

## File Organization 
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

## Citation
```
@misc{caribbean_roof_detection_2023,
  title={Fusing VHR Post-disaster Aerial Imagery and LiDAR Data for Roof Classification in the Caribbean},
  author={Tingzon, Isabelle and Cowan, Nuala Margaret and Chrzanowski, Pierre},
  journal={arXiv preprint arXiv:2307.16177},
  year={2023},
}
```
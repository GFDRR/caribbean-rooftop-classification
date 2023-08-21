<div align="center">

# Digital Earth for a Resilient Caribbean 
Data Fusion of Post-disaster Aerial Imagery and LiDAR Data for Rooftop Classification using CNNs

<p>
<b><a href="#-description">Description</a></b>
|
<b><a href="#-datasets">Datasets</a></b>
|
<b><a href="#-methods">Methods</a></b>
|
<b><a href="#-file-organization">File Organization</a></b>
|
<b><a href="#citation">Citation</a></b>
</p>

</div>

## 📜 Description
This work leverages convolutional neural networks (CNNs) and data fusion techniques for the automated classification of rooftop  characteristics from very high-resolution orthophotos and airborne LiDAR data. 

The Digital Earth for a Resilient Caribbean project aims help governments produce more timely building information to improve resilience and disaster response in the Caribbean.
<p>
<img src="./assets/results.png" width="70%" height="70%" />

## 📂 Datasets
To generate our ground truth dataset, we used the following three data sources for Dominica and St. Lucia: (1) VHR aerial imagery, (2) LiDAR data, and (3) building footprints in the form of georeferenced vector polygons.

### Roof Characteristics
We annotated a total of 8,345 buildings according to two attributes: (1) roof type and (2) roof material. The following figures below illustrate examples of the RGB orthophoto and LiDAR-derived image patches.
<p>
<img src="./assets/roof-characteristics.png" width="70%" height="70%" />

## Citation
Use this bibtex to cite this repository:
```
@misc{caribbean_roof_detection_2023,
  title={Fusing VHR Post-disaster Aerial Imagery and LiDAR Data for Roof Classification in the Caribbean},
  author={Tingzon, Isabelle and Cowan, Nuala Margaret and Chrzanowski, Pierre},
  journal={arXiv preprint arXiv:2307.16177},
  year={2023},
}
```
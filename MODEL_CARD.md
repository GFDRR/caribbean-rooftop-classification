# Model Card

These are CNN models trained following the method described in the paper:
"Mapping Housing Stock Characteristics from Drone Images for Climate Resilience in the Caribbean".

We provide 3 models: 
- 1 EfficientNet-B0 model for roof type classification trained on a combination of data from Dominica and Saint Lucia.
	- Classes: Gable, Hip, Flat, No Roof
	- See: [issatingzon/cnn-roof_type-efficientnetb0-RGB_DOM_LCA](https://huggingface.co/issatingzon/cnn-roof_type-efficientnetb0-RGB_DOM_LCA)
- 1 ResNet-50 model for roof material classification trained on Dominica data.
	- Classes: Healthy Metal, Irregular Metal, Concrete/Cement, Incomplete, Blue Tarpaulin
	- See: [issatingzon/cnn-roof_material-resnet50-RGB_DOM](https://huggingface.co/issatingzon/cnn-roof_material-resnet50-RGB_DOM)
- 1 EfficientNet-B0 model for roof material classification trained on Saint Lucia data. 
	- Classes: Healthy Metal, Irregular Metal, Concrete/Cement, Incomplete
	- See: [issatingzon/cnn-roof_type-efficientnetb0-RGB_DOM_LCA](https://huggingface.co/issatingzon/cnn-roof_type-efficientnetb0-RGB_DOM_LCA)

## Model Details
- Roof type and roof material classification models, developed under the Digital Earth for Resilient Caribbean, a World Bank Project funded by GFDRR, 2023, v1.
- Convolutional Neural Networks, pre-trained on the Imagenet dataset and fine-tuned for roof type and roof material classification.

## Model Description

- **Developed by:** The World Bank, GFDRR
- **Model type:** Convolutional Neural Networks (CNNs)
- **License:** MIT License

- **Repository:** https://github.com/GFDRR/caribbean-rooftop-classification
- **Papers:** https://arxiv.org/pdf/2307.16177, https://arxiv.org/pdf/2312.10306

## How to Get Started with the Model

We recommend following the two tutorials to get started with the models:
- Part 1: [Building Footprint Delineation for Disaster Risk Reduction and Response (DRR)](https://colab.research.google.com/github/GFDRR/caribbean-rooftop-classification/blob/master/tutorials/01_building_delineation.ipynb)
- Part 2: [Rooftop Type and Roof Material Classification using Drone Imagery](https://colab.research.google.com/github/GFDRR/caribbean-rooftop-classification/blob/master/tutorials/02_building_classification.ipynb)

## Intended Use
- Intended to be used for the rapid assessment of housing stock characteristics and building damage in post-disaster contexts on a building-level, specifically in Dominica and Saint Lucia.
- The work is developed in support of climate resilience initiatives such as the Resilient Housing Scheme 2030 in Dominica, which strives to make 90% of Dominica’s housing stock resilient by 2030.

## Factors
- Potential relevant factors include the geographic landscape and unique building characteristics within the country of interest; degree of urbanization (e.g. urban, semi-urban, rural); socioeconomic factors; spatial resolution of aerial images; and disaster context (pre- or post-disaster).

## Metrics
- Evaluation metrics include macro-averaged F1-score, precision, recall, and accuracy. For more information, see Section F of the Appendix in this [paper](https://arxiv.org/pdf/2307.16177.pdf).
- All metrics reported at the 0.5 decision threshold. 

## Training Data and Evaluation Data
- Manually annotated approximately 15,000 rooftops across Dominica and Saint Lucia via visual inspection of very high-resolution aerial images (e.g. drone images, RGB orthophotos, LiDAR data).
- Within each country, we split the data into designated training and test sets, using stratified random sampling to preserve the percentage of samples for each class. We also evaluate the cross-country generalizability of the models as well as the performance of a model trained using the combination of the two training datasets ("combined" model).

## Evaluation
We refer users to the associated papers for the evaluation results.


## Ethical Considerations
- The use of the model results may lead to unintended consequences, e.g. 
	- The identification of less developed settlements, based on rooftop characteristics, may lead to fewer investments in the area, causing further deprivation. Consider the local context when applying the model to avoid misuse. 
	- Group privacy issues arising from the extraction of group-level information from aerial images.
- When considering the DRM application, we note that vulnerability relates only to the characteristics detected in the aerial imagery.

## Caveats and Recommendations
- Did not evaluate across different socioeconomic groups or degrees of urbanization. Further work is needed to evaluate the model across a range of socioeconomic groups.
- We urge caution in applying the models off-the-shelf to new geographic contexts (e.g. countries outside of the Caribbean) as the initial results of cross-country cross-validation indicate performance degradation in the face of geographic distribution shift. 
- The training data for Dominica is derived from VHR post-disaster aerial images taken in the aftermath of Category 5 Hurricane Maria in 2018-2019. Class distributions may have shifted since this period, e.g. the training data consists of 15% blue tarpaulins (which are used to cover severely damaged rooftops), but these blue tarpaulins may not be as prevalent in more recent years due to reparation, reconstruction, and retrofitting initiatives in Dominica. 



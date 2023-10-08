import os
import random
import subprocess
import rasterio as rio
import rasterio.mask

from tqdm import tqdm
import geopandas as gpd
import pandas as pd

import matplotlib.pyplot as plt
from rasterio.plot import show

from shapely.geometry.polygon import Polygon
from shapely.geometry import box
import logging

logging.basicConfig(level=logging.INFO)
pd.set_option("mode.chained_assignment", None)

SEED = 42


def get_classes_dict(attribute):
    classes_dict = {
        "roof_material": {
            0: 'INCOMPLETE',
            1: 'HEALTHY_METAL', 
            2: 'IRREGULAR_METAL',
            3: 'CONCRETE_CEMENT',  
            4: 'BLUE_TARP',
        },
        "roof_type": {
            0: 'NO_ROOF',
            1: 'GABLE', 
            2: 'HIP', 
            3: 'FLAT'
        },
    }
    return classes_dict[attribute]


def remove_ticks(ax):
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_axis_off()


def read_image(filepath, n_channels=3):
    image = rio.open(filepath)
    if n_channels == 3:
        image = image.read([1, 2, 3])
    elif n_channels == 1:
        image = image.read([1]) * 255
    image[image == 0] = 255
    return image


def get_annotated_bldgs(config, index):
    bldgs_path = os.path.join(config["vectors_dir"], config["bldgs_files"][index])
    bldgs = gpd.read_file(bldgs_path)

    if "roof_type" in bldgs.columns and "roof_material" in bldgs.columns:
        bldgs = bldgs[(bldgs.roof_type != "OTHER") & (bldgs.roof_material != "OTHER")]
        bldgs.roof_type = bldgs.roof_type.replace({"PYRAMID": "HIP", "HALF_HIP": "HIP"})
        bldgs = bldgs[(~bldgs.roof_type.isna()) | (~bldgs.roof_material.isna())]
        logging.info(f"Dimensions (non-null): {bldgs.shape}")
        logging.info(bldgs.roof_material.value_counts())
        logging.info(bldgs.roof_type.value_counts())

    return bldgs


def inspect_image_crops(
    rgb_data,
    column,
    value,
    index=0,
    n_rows=5,
    n_cols=5,
    figsize=(15, 10),
    lidar_data=[],
    prefix=''
):
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=figsize)
    rgb_data = rgb_data[rgb_data[column] == value].reset_index()
    samples = rgb_data.iloc[index : index + ((n_rows  * 2) * n_cols)].iterrows()
    row_index, col_index = 0, 0

    increment = 1
    for _, item in samples:
        rgb_filepath = prefix + item["filepath"]
        
        rgb_image = read_image(rgb_filepath, n_channels=3)
        show(rgb_image, ax=axes[row_index, col_index])
        remove_ticks(axes[row_index, col_index])
        
        lidar_flag = False
        if len(lidar_data) > 0:
            filename = item['filename']
            if filename in lidar_data['filename'].unique():
                lidar_item = lidar_data[lidar_data["filename"] == filename]
                lidar_filepath = prefix + lidar_item.iloc[0].filepath
                lidar_image = read_image(lidar_filepath, n_channels=1)
                show(lidar_image, ax=axes[row_index + 1, col_index])
                remove_ticks(axes[row_index + 1, col_index])
                lidar_flag = True

        axes[row_index, col_index].set_title(
            f"{item.aoi}_{item.UID}", 
            fontdict={"fontsize": 9}
        )

        col_index += 1
        if col_index >= n_cols:
            if lidar_flag:
                row_index += 2
            else:
                row_index += 1
            col_index = 0
            
        if row_index >= n_rows*2:
            break
            


def visualize_image_crops(
    data, column, n_samples=8, n_channels=3, figsize=(15, 10), prefix='', title=True
):
    categories = data[column].unique()
    fig, axs = plt.subplots(len(categories), 1, figsize=figsize)
    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    for subfig, category in zip(subfigs, categories):
        subfig.suptitle(f"{category}")
        axes = subfig.subplots(nrows=1, ncols=n_samples)
        subdata = data[data[column] == category]
        n_samples = min(n_samples, len(subdata))
        samples = subdata.sample(n_samples).reset_index(drop=True)

        for i, item in samples.iterrows():
            filename = prefix + item['filepath']
            image = read_image(filename, n_channels)
            show(image, ax=axes[i])
            if title:
                title = f"{item.aoi}_{item.UID}"
                axes[i].set_title(title, fontdict={"fontsize": 9})
            remove_ticks(axes[i])


def crop_shape(shape, scale, in_file, out_file):
    shape.geometry = shape.geometry.apply(lambda x: x.minimum_rotated_rectangle)
    shape.geometry = shape.geometry.scale(scale, scale)

    with rio.open(in_file) as src:
        out_image, out_transform = rasterio.mask.mask(
            src, [shape.iloc[0].geometry], crop=True
        )
        out_meta = src.meta
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
    with rio.open(out_file, "w", **out_meta) as dest:
        dest.write(out_image)


def generate_image_crops(data, in_file, out_path, aoi, scale=1.5, clip=False):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    pbar = tqdm(
        enumerate(data.iterrows()),
        total=len(data),
        bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}",
    )
    
    csv = []
    for index, (_, row) in pbar:
        uid = int(row["UID"])
        roof_type = row['roof_type']
        roof_material = row['roof_material']
        
        filename = f"{aoi}-{uid}-{roof_type}-{roof_material}.tif"
        out_file = os.path.join(out_path, filename)
        shape = data[(data.UID == uid)]
        crop_shape(shape, scale, in_file, out_file)

        if clip:
            with rio.open(out_file) as image:
                meta = image.meta
                array = image.read(1)
                array[array < 0] = 0
            with rio.open(out_file, "w", **meta) as dst:
                dst.write_band(1, array)
        
        image_src = in_file.split('/')[-2].upper()
        csv.append([uid, aoi, out_file, filename, image_src, roof_type, roof_material])
        
    csv = pd.DataFrame(csv, columns=[
        "UID",
        "aoi", 
        "filepath",
        "filename",
        "image_src", 
        "roof_type", 
        'roof_material'
    ])
    return csv


def close_holes(poly: Polygon) -> Polygon:
    """
    Close polygon holes by limitation to the exterior ring.
    Source: https://stackoverflow.com/a/61466689

    Args:
        poly: Input shapely Polygon
    Example:
        df.geometry.apply(lambda p: close_holes(p))
    """
    if poly.interiors:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly


def show_crop(image, shape, title=""):
    """Crops an image based on the polygon shape.
    Reference: https://rio.readthedocs.io/en/latest/api/rio.mask.html#rio.mask.mask

    Args:
        image (str): Image file path (.tif)
        shape (geometry): The tile with which to crop the image
        title(str): Image title
    """

    with rio.open(image) as src:
        out_image, out_transform = rio.mask.mask(src, shape, crop=True)
        show(out_image, title=title)


def merge_polygons(gpkg_dir, crs, max_area, min_area, tolerance):
    files = next(os.walk(gpkg_dir), (None, None, []))[2]

    polygons = []
    for file in files:
        if file.split(".")[-1] == "gpkg":
            filepath = os.path.join(gpkg_dir, file)
            gdf = gpd.read_file(filepath)
            gdf = gdf.set_crs(crs, allow_override=True)
            polygons.append(gdf)

    polygons = pd.concat(polygons)
    polygons = gpd.GeoDataFrame(polygons)[["geometry"]]

    geoms = polygons.geometry.unary_union
    polygons = gpd.GeoDataFrame(geometry=[geoms], crs=crs)
    polygons = polygons.explode().reset_index(drop=True)
    polygons.geometry = polygons.geometry.apply(lambda p: close_holes(p))

    polygons = polygons.to_crs("EPSG:32620")
    polygons["area"] = polygons.geometry.area
    polygons = polygons[(polygons.area < max_area) & (polygons.area > min_area)]

    polygons = polygons.to_crs("EPSG:4326")
    polygons.geometry = polygons.geometry.simplify(tolerance=tolerance)
    polygons = polygons.to_crs(crs)

    return polygons


def generate_tiles(image_file, size=3000):
    """Generates n x n polygon tiles.

    Args:
      image_file (str): Image file path (.tif)
      output_file (str): Output file path (.geojson)
      area_str (str): Name of the region
      size(int): Window size

    Returns:
      GeoPandas DataFrame: Contains 64 x 64 polygon tiles
    """

    # Open the raster image using rio
    with rio.open(image_file) as raster:
        width, height = raster.shape

    # Create a dictionary which will contain our 64 x 64 px polygon tiles
    # Later we'll convert this dict into a GeoPandas DataFrame.
    geo_dict = {"geometry": []}
    index = 0

    # Do a sliding window across the raster image
    with tqdm(total=width * height) as pbar:
        for w in range(0, width, int(size / 2)):
            for h in range(0, height, int(size / 2)):
                # Create a Window of your desired size
                window = rio.windows.Window(h, w, size, size)
                # Get the georeferenced window bounds
                bbox = rio.windows.bounds(window, raster.transform)
                # Create a shapely geometry from the bounding box
                bbox = box(*bbox)

                # Update dictionary
                # geo_dict['id'].append(uid)
                geo_dict["geometry"].append(bbox)

                index += 1
                pbar.update(size * size)

    # Cast dictionary as a GeoPandas DataFrame
    results = gpd.GeoDataFrame(pd.DataFrame(geo_dict))
    # Set CRS to EPSG:4326
    results.crs = {"init": "epsg:4326"}
    return results


def generate_train_test(
    data,
    out_dir,
    test_size,
    attributes=None,
    test_aoi=None,
    test_src=None,
    stratified=True,
    verbose=True,
):
    data["dataset"] = None
    total_size = len(data)
    test_size = int(total_size * test_size)
    logging.info(f"Data dimensions: {total_size}")

    test = data.copy()
    if test_aoi != None:
        test = data[data.aoi == test_aoi]
    if test_src != None:
        test = data[data.image_src == test_src]
    if stratified:
        value_counts = data.groupby(attributes)[attributes[-1]].value_counts()
        value_counts = pd.DataFrame(value_counts).reset_index()
        for _, row in value_counts.iterrows():
            subtest = test.copy()
            for i in range(len(attributes)):
                subtest = subtest[subtest[attributes[i]] == row[attributes[i]]]
            subtest_size = int(test_size * (row['count'] / total_size))
            if subtest_size > len(subtest):
                subtest_size = len(subtest)
            subtest_files = subtest.sample(
                subtest_size, random_state=SEED
            ).filename.values
            in_test = data["filename"].isin(subtest_files)
            data.loc[in_test, "dataset"] = "TEST"
    data.dataset = data.dataset.fillna("TRAIN")

    if verbose:
        value_counts["percentage"] = value_counts["count"]/total_size
        logging.info(value_counts)
        subcounts = pd.DataFrame(
            data.groupby(attributes + ["dataset"]).size().reset_index()
        )
        subcounts.columns = attributes + ["dataset", "count"]
        subcounts["percentage"] = (
            subcounts[subcounts.dataset == "TEST"]["count"] / test_size
        )
        subcounts = subcounts.set_index(attributes + ["dataset"])
        logging.info(subcounts)
        for attribute in attributes:
            logging.info(data[attribute].value_counts())
        logging.info(data.image_src.value_counts())
        logging.info(data.dataset.value_counts())

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    return data

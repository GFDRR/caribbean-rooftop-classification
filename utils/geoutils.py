import os
import random
import subprocess
import rasterio as rio

import geopandas as gpd
import pandas as pd

import matplotlib.pyplot as plt
from rasterio.plot import show
from tqdm import tqdm

import logging
logging.basicConfig(level = logging.INFO)
pd.set_option('mode.chained_assignment', None)


def get_classes_dict(attribute):
    classes_dict = {
        "roof_material": [
            "BLUE_TARP",
            "CONCRETE_CEMENT",
            "HEALTHY_METAL",
            "INCOMPLETE",
            "IRREGULAR_METAL",
        ],
        "roof_type": ["FLAT", "GABLE", "HIP", "NO_ROOF"],
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


def get_annotated_bldgs(config):
    bldgs_path = os.path.join(config["vectors_dir"], config["version"], config["bldgs_file"])
    bldgs = gpd.read_file(bldgs_path)

    if "roof_type" in bldgs.columns and "roof_material" in bldgs.columns:
        bldgs = bldgs[(bldgs.roof_type != "OTHER") & (bldgs.roof_material != "OTHER")]
        bldgs.roof_type = bldgs.roof_type.replace({"PYRAMID": "HIP", "HALF_HIP": "HIP"})
        logging.info(f"Dimensions: {bldgs.shape}")

        bldgs_shape = bldgs[
            (~bldgs.roof_type.isna()) | (~bldgs.roof_material.isna())
        ].shape
        logging.info(f"Dimensions (non-null): {bldgs_shape}")
        logging.info(bldgs.roof_material.value_counts())
        logging.info(bldgs.roof_type.value_counts())

    return bldgs


def get_image_dirs(config):
    data_dir = os.path.join(config["rasters_dir"], "tiles", config["version"])
    rgb_path = os.path.join(data_dir, 'RGB')
    lidar_path = os.path.join(data_dir, "LIDAR")
    return rgb_path, lidar_path


def inspect_image_crops(
    data,
    column,
    value,
    rgb_path,
    lidar_path,
    aoi,
    index=0,
    n_rows=5,
    n_cols=5,
    figsize=(15, 10),
):
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=figsize)
    data = data[data[column] == value]
    samples = data.iloc[index : index + (n_rows * n_cols)].iterrows()
    row_index, col_index = 0, 0

    for _, item in samples:
        filename = f"{aoi}_{item.UID}.tif"
        lidar_filepath = os.path.join(lidar_path, column, value, filename)
        rgb_filepath = os.path.join(rgb_path, column, value, filename)

        if os.path.isfile(lidar_filepath) and os.path.isfile(rgb_filepath):
            lidar_image = read_image(lidar_filepath, n_channels=1)
            rgb_image = read_image(rgb_filepath, n_channels=3)

            show(rgb_image, ax=axes[row_index, col_index])
            show(lidar_image, ax=axes[row_index + 1, col_index])

            remove_ticks(axes[row_index, col_index])
            remove_ticks(axes[row_index + 1, col_index])

            axes[row_index, col_index].set_title(
                int(item.UID), fontdict={"fontsize": 9}
            )

            col_index += 1

            if col_index >= n_cols:
                row_index += 2
                col_index = 0


def visualize_image_crops(
    folder_path, column, n_samples=8, n_channels=3, figsize=(15, 10)
):
    out_dir = os.path.join(folder_path, column)
    folders = [folder.name for folder in os.scandir(out_dir)]
    fig, axs = plt.subplots(len(folders), 1, figsize=figsize)
    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    for row, (folder, subfig) in enumerate(zip(folders, subfigs)):
        out_path = os.path.join(out_dir, folder)
        files = os.listdir(out_path)

        subfig.suptitle(f"{folder}")
        axes = subfig.subplots(nrows=1, ncols=n_samples)
        n_samples = min(n_samples, len(files))
        samples = random.sample(files, n_samples)

        for i, file in enumerate(samples):
            out_file = os.path.join(out_path, file)
            image = read_image(out_file, n_channels)
            show(image, ax=axes[i])
            axes[i].set_title(file, fontdict={"fontsize": 9})
            remove_ticks(axes[i])


def crop_shape(shape, filename, scale, in_file, out_file):
    shape.geometry = shape.geometry.apply(lambda x: x.minimum_rotated_rectangle)
    shape.geometry = shape.geometry.scale(scale, scale)
    shape.to_file(filename, driver="GPKG")
    subprocess.call(
        f"gdalwarp -q -cutline {filename} -crop_to_cutline -dstalpha {in_file} {out_file}",
        shell=True,
    )


def generate_image_crops(data, column, in_file, out_dir, aoi, scale=1.5, clip=False):
    logging.info(f"{column} size: {data[~data[column].isna()].shape}")
    
    data = data[(~data[column].isna())]
    for attr in data[column].unique():
        out_path = os.path.join(out_dir, attr)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        subset = data[data[column] == attr]
        pbar = tqdm(
            enumerate(subset.iterrows()),
            total=len(subset),
            bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}",
        )
        for index, (_, row) in pbar:
            uid = row["UID"]
            out_file = os.path.join(out_path, f"{aoi}_{uid}.tif")
            shape = data[(data.UID == uid)]
            filename = "temp.gpkg"
            crop_shape(shape, filename, scale, in_file, out_file)

            if clip:
                with rio.open(out_file) as image:
                    meta = image.meta
                    array = image.read(1)
                    array[array < 0] = 0
                with rio.open(out_file, "w", **meta) as dst:
                    dst.write_band(1, array)

            pbar.set_description(f"{attr}")

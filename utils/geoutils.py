import os
import random
import subprocess
import rasterio as rio

import matplotlib.pyplot as plt
from rasterio.plot import show
from tqdm import tqdm


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


def inspect_image_crops(
    data,
    column,
    value,
    rgb_path,
    ndsm_path,
    iso,
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
        filename = f"{iso}_{item.UID}.tif"
        ndsm_filepath = os.path.join(ndsm_path, column, value, filename)
        ndsm_image = read_image(ndsm_filepath, n_channels=1)

        rgb_filepath = os.path.join(rgb_path, column, value, filename)
        rgb_image = read_image(rgb_filepath, n_channels=3)

        show(rgb_image, ax=axes[row_index, col_index])
        show(ndsm_image, ax=axes[row_index + 1, col_index])

        remove_ticks(axes[row_index, col_index])
        remove_ticks(axes[row_index + 1, col_index])

        axes[row_index, col_index].set_title(int(item.UID), fontdict={"fontsize": 9})

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
        f"gdalwarp -cutline {filename} -crop_to_cutline -dstalpha {in_file} {out_file}",
        shell=True,
    )

    
def generate_image_crops(data, column, in_file, out_dir, iso, scale=1.5, clip=False):
    print(f"{column} size: {data[~data[column].isna()].shape}")
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
            out_file = os.path.join(out_path, f"{iso}_{uid}.tif")
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

import os
import sys

from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np

from PIL import Image
from shapely.geometry import shape
from rasterio.features import shapes

import torch
import torch.nn.functional as nn

sys.path.insert(0, "./utils/")
import cnn_utils
import geoutils
import config
import sam_utils

import logging
logging.basicConfig(level=logging.INFO)


def predict(bldgs, in_file, out_dir, exp_config, model_file=None, prefix=""):
    c = config.create_config(exp_config, prefix=prefix)
    classes = geoutils.get_classes_dict(c["attribute"])
    logging.info(f"Config: {c}")

    if not model_file:
        exp_dir = os.path.join(c["exp_dir"], c["version"], c["exp_name"])
        model_file = os.path.join(exp_dir, "best_model.pth")

    model = load_model(c, model_file, n_classes=len(classes))
    return generate_predictions(bldgs, model, c, in_file, out_dir, classes)


def load_model(c, model_file, n_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cnn_utils.get_model(c["model"], n_classes, c["mode"], c["dropout"])
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()
    logging.info("Model file {} successfully loaded.".format(model_file))
    return model


def generate_predictions(data, model, c, in_file, out_dir, classes, scale=1.5):
    preds, probs = [], []
    pbar = tqdm(
        enumerate(data.iterrows()),
        total=len(data),
        bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}",
    )
    for index, (_, row) in pbar:
        out_file = os.path.join(out_dir, "temp.tif")
        shape = data[(data.UID == row["UID"])]
        out_shape = os.path.join(out_dir, "temp.gpkg")
        if os.path.exists(out_file):
            os.remove(out_file)
        geoutils.crop_shape(shape, out_shape, scale, in_file, out_file)

        if os.path.exists(out_file):
            image = cnn_utils.read_image(out_file, c["mode"])
            transforms = cnn_utils.get_transforms(c["img_size"], c["mode"])
            output = model(transforms["test"](image).unsqueeze(0))
            probs.append(nn.softmax(output, dim=1).detach().numpy()[0])
            _, pred = torch.max(output, 1)
            label = str(classes[int(pred[0])])
            preds.append(label)
        else:
            probs.append([np.nan] * len(classes))
            preds.append(np.nan)

    probs_col = [f"{class_}_PROB" for class_ in classes]
    probs = pd.DataFrame(probs, columns=probs_col)
    data[c["attribute"]] = preds
    return gpd.GeoDataFrame(pd.concat([data, probs], axis=1))


def model_predict(
    image_file,
    model,
    text_prompt,
    box_threshold=0.3,
    text_threshold=0.3,
    out_file=None,
    visualize=True,
):
    with rio.open(image_file) as src:
        image_np = src.read().transpose((1, 2, 0))
        transform = src.transform
        crs = src.crs
    image_pil = Image.fromarray(image_np[:, :, :3])

    masks, boxes, phrases, logit = model.predict(
        image_pil,
        text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    torch.cuda.empty_cache()

    # Generate mask overlay
    mask_overlay = np.zeros_like(image_np[..., 0], dtype=np.uint8)
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy().astype(np.uint8)
        mask_overlay += ((mask > 0) * (i + 1)).astype(np.uint8)
    mask_overlay = (mask_overlay > 0) * 255

    if out_file:
        save_predictions(out_file, mask_overlay, transform, crs)

    if visualize:
        visualize_predictions(image_pil, boxes, mask_overlay)


def visualize_predictions(image_pil, boxes, mask_overlay):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_pil)

    for box in boxes:
        box = box.cpu().numpy()
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        plt.gca().add_patch(rect)

    plt.imshow(mask_overlay, cmap="viridis", alpha=0.4)
    plt.title(f"Segmented")
    plt.show()

    return mask_overlay


def save_predictions(filename, mask_overlay, transform, crs):
    mask = mask_overlay.astype("int16")  
    results = (
        {"properties": {"raster_val": v}, "geometry": s}
        for i, (s, v) in enumerate(shapes(mask, transform=transform))
        if v == 255  
    )

    geoms = list(results)
    if len(geoms) > 0:
        gdf = gpd.GeoDataFrame.from_features(geoms)
        gdf.crs = crs 
        gdf.to_file(filename, driver="GPKG")


def predict_image_crop(
    image, text_prompt, shape, model, out_dir, uid, box_threshold, text_threshold
):
    with rio.open(image) as src:
        out_image, out_transform = rio.mask.mask(src, shape, crop=True)

        if (np.mean(out_image) == 255) or (np.mean(out_image) == 0):
            return

        # Get the metadata of the source image and update it
        # with the width, height, and transform of the cropped image
        out_meta = src.meta
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        # Save the cropped image as a temporary TIFF file.
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        temp_tif = os.path.join(out_dir, f"{uid}.tif")
        with rio.open(temp_tif, "w", **out_meta) as dest:
            dest.write(out_image)

        out_file = os.path.join(out_dir, f"{uid}.gpkg")
        model_predict(
            temp_tif,
            model,
            text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            out_file=out_file,
            visualize=False,
        )


def predict_image(
    image_file,
    out_dir,
    out_file,
    text_prompt,
    tile_size=3000,
    box_threshold=0.3,
    text_threshold=0.3,
    max_area=1000,
    min_area=5,
    tolerance=0.000005,
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    model = sam_utils.LangSAM()
    tiles = geoutils.generate_tiles(image_file, size=tile_size)
    with rio.open(image_file) as src:
        crs = src.crs

    for index in tqdm(range(len(tiles)), total=len(tiles)):
        shape = [tiles.iloc[index]["geometry"]]
        predict_image_crop(
            image_file,
            text_prompt,
            shape,
            model,
            out_dir,
            index,
            box_threshold,
            text_threshold,
        )

    polygons = geoutils.merge_polygons(out_dir, crs, max_area, min_area, tolerance)
    polygons.to_file(out_file, driver="GPKG")
    return polygons

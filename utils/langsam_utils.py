import os
import numpy as np
import torch
import rasterio as rio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import groundingdino.datasets.transforms as T

from PIL import Image
import rasterio.mask
from rasterio.plot import show
from matplotlib.patches import Rectangle
from shapely.geometry import box

from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download

from segment_anything import sam_model_registry
from segment_anything import SamPredictor

from shapely.geometry.polygon import Polygon
from shapely.geometry import shape
from rasterio.features import shapes

from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import itertools


# Define constants
SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

# Default cache path for model checkpoints
CACHE_PATH = os.environ.get(
    "TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints")
)


# Class definition for LangSAM
class LangSAM:
    """
    A Language-based Segment-Anything Model (LangSAM) class which combines GroundingDINO and SAM.
    """

    def __init__(self, sam_type: str = "vit_h"):
        """
        Initialize the LangSAM instance.

        Parameters:
        sam_type (str): Type of SAM model to use. Default is "vit_h".
        """
        if sam_type not in SAM_MODELS:
            raise ValueError(
                f"Invalid SAM model type. Available options are {list(SAM_MODELS.keys())}."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_groundingdino()
        self.build_sam(sam_type)

    def build_sam(self, sam_type: str):
        """
        Build the SAM model.

        Parameters:
        sam_type (str): Type of SAM model to use.
        """
        checkpoint_url = SAM_MODELS[sam_type]
        sam = sam_model_registry[sam_type]()
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
        sam.load_state_dict(state_dict, strict=True)
        sam.to(device=self.device)
        self.sam = SamPredictor(sam)

    def build_groundingdino(self):
        """
        Build the GroundingDINO model.
        """
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(
            ckpt_repo_id, ckpt_filename, ckpt_config_filename, self.device
        )

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        """
        Run the GroundingDINO model prediction.

        Parameters:
        image_pil (Image): Input PIL Image.
        text_prompt (str): Text prompt for the model.
        box_threshold (float): Box threshold for the prediction.
        text_threshold (float): Text threshold for the prediction.

        Returns:
        Tuple containing boxes, logits, and phrases.
        """
        image_trans = transform_image(image_pil)
        boxes, logits, phrases = predict(
            model=self.groundingdino,
            image=image_trans,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H]).to(
            boxes.device
        )
        # Ensure tensor is on the same device
        return boxes, logits, phrases

    def predict_sam(self, image_pil: Image, boxes: torch.Tensor):
        """
        Run the SAM model prediction.

        Parameters:
        image_pil (Image): Input PIL Image.
        boxes (torch.Tensor): Tensor of bounding boxes.

        Returns:
        Masks tensor.
        """
        image_array = np.array(image_pil)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(
            boxes, image_array.shape[:2]
        )
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()

    def predict(
        self,
        image_pil: Image,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
    ):
        """
        Run both GroundingDINO and SAM model prediction.

        Parameters:
        image_pil (Image): Input PIL Image.
        text_prompt (str): Text prompt for the model.
        box_threshold (float): Box threshold for the prediction.
        text_threshold (float): Text threshold for the prediction.

        Returns:
        Tuple containing masks, boxes, phrases, and logits.
        """
        boxes, logits, phrases = self.predict_dino(
            image_pil, text_prompt, box_threshold, text_threshold
        )
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, phrases, logits


# Define helper functions
def load_model_hf(
    repo_id: str, filename: str, ckpt_config_filename: str, device: str = "cpu"
) -> torch.nn.Module:
    """
    Loads a model from HuggingFace Model Hub.

    Parameters:
    repo_id (str): Repository ID on HuggingFace Model Hub.
    filename (str): Name of the model file in the repository.
    ckpt_config_filename (str): Name of the config file for the model in the repository.
    device (str): Device to load the model onto. Default is 'cpu'.

    Returns:
    torch.nn.Module: The loaded model.
    """
    # Ensure the repo ID and filenames are valid
    assert isinstance(repo_id, str) and repo_id, "Invalid repository ID"
    assert isinstance(filename, str) and filename, "Invalid model filename"
    assert (
        isinstance(ckpt_config_filename, str) and ckpt_config_filename
    ), "Invalid config filename"

    # Download the config file and build the model from it
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    model.to(device)

    # Download the model checkpoint and load it into the model
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()

    return model


def transform_image(image: Image) -> torch.Tensor:
    """
    Transforms an image using standard transformations for image-based models.

    Parameters:
    image (Image): The PIL Image to be transformed.

    Returns:
    torch.Tensor: The transformed image as a tensor.
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image, None)
    return image_transformed


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
    # Save the individual segmentations into a multi-part ShapeFile
    mask = mask_overlay.astype("int16")  # Convert the mask to integer type
    results = (
        {"properties": {"raster_val": v}, "geometry": s}
        for i, (s, v) in enumerate(shapes(mask, transform=transform))
        if v == 255  # Add condition to only keep 'trees'
    )

    geoms = list(results)
    if len(geoms) > 0:
        gdf = gpd.GeoDataFrame.from_features(geoms)
        gdf.crs = crs  # Assign the Coordinate Reference System of the original image

        # Save to file
        gdf.to_file(filename, driver="GPKG")


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


def predict_image_crop(
    image, text_prompt, shape, model, out_dir, uid, box_threshold, text_threshold
):
    """Generates model prediction using trained model

    Args:
      image (str): Image file path (.tiff)
      shape (geometry): The tile with which to crop the image
      classes (list): List of LULC classes

    Return
      str: Predicted label
    """

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


def merge_polygons(gpkg_dir, crs, max_area, min_area, tolerance):
    files = filenames = next(os.walk(gpkg_dir), (None, None, []))[2]

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
    polygons['area'] = polygons.geometry.area
    polygons = polygons[(polygons.area < max_area) & (polygons.area > min_area)]
    
    polygons = polygons.to_crs("EPSG:4326")
    polygons.geometry = polygons.geometry.simplify(tolerance=tolerance)
    polygons = polygons.to_crs(crs)
        
    return polygons


def predict_image(
    image_file,
    text_prompt,
    tiles,
    model,
    out_dir,
    out_file,
    box_threshold=0.3,
    text_threshold=0.3,
    max_area=1000, 
    min_area=1,
    tolerance=0.000005
):
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
        
    polygons = merge_polygons(out_dir, crs, max_area, min_area, tolerance)
    polygons.to_file(out_file, driver="GPKG")
    return polygons

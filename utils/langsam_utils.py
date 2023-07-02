import os
import numpy as np
import torch
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import groundingdino.datasets.transforms as T

from PIL import Image
from rasterio.plot import show
from matplotlib.patches import Rectangle

from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download

from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from shapely.geometry import shape
from rasterio.features import shapes

# Define constants
SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

# Default cache path for model checkpoints
CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints"))

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
            raise ValueError(f"Invalid SAM model type. Available options are {list(SAM_MODELS.keys())}.")

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
        self.groundingdino = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, self.device)

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
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H]).to(boxes.device)  # Ensure tensor is on the same device
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
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()

    def predict(self, image_pil: Image, text_prompt: str, box_threshold: float, text_threshold: float):
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
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, phrases, logits
    
    
# Define helper functions
def load_model_hf(repo_id: str, filename: str, ckpt_config_filename: str, device: str = 'cpu') -> torch.nn.Module:
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
    assert isinstance(ckpt_config_filename, str) and ckpt_config_filename, "Invalid config filename"

    # Download the config file and build the model from it
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    model.to(device)

    # Download the model checkpoint and load it into the model
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
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
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image, None)
    return image_transformed

def get_image_arrays(image_file):
    with rasterio.open(image_file) as src:
        image_np = src.read().transpose((1, 2, 0))  # Convert rasterio image to numpy array
        transform = src.transform  # Save georeferencing information
        crs = src.crs  # Save the Coordinate Reference System

    image_pil = Image.fromarray(image_np[:, :, :3])
    return image_np, image_pil, transform

    
def visualize(image_np, image_pil, boxes, masks):
    mask_overlay = np.zeros_like(image_np[..., 0], dtype=np.uint8)  # Adjusted for single channel

    for i, (box, mask) in enumerate(zip(boxes, masks)):
        # Convert tensor to numpy array if necessary and ensure it contains integers
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy().astype(np.uint8)  # If mask is on GPU, use .cpu() before .numpy()
        mask_overlay += ((mask > 0) * (i + 1)).astype(np.uint8)  # Assign a unique value for each mask

    # Normalize mask_overlay to be in [0, 255]
    mask_overlay = (mask_overlay > 0) * 255  # Binary mask in [0, 255]

    # Display the original image with all mask overlays and bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image_pil)

    for box in boxes:
        # Draw bounding box
        box = box.cpu().numpy()  # Convert the tensor to a numpy array
        rect = patches.Rectangle(
            (box[0], box[1]), 
            box[2] - box[0], 
            box[3] - box[1], 
            linewidth=1, 
            edgecolor='r', 
            facecolor='none'
        )
        plt.gca().add_patch(rect)

    plt.imshow(mask_overlay, cmap='viridis', alpha=0.4)  # Overlay the mask with some transparency
    plt.title(f"Segmented")
    plt.show()
    
    return mask_overlay
   
    
def save_predictions(filename, mask_overlay, transform):
    # Save the individual segmentations into a multi-part ShapeFile
    mask = mask_overlay.astype('int16')  # Convert the mask to integer type
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(shapes(mask, transform=transform))
        if v == 255  # Add condition to only keep 'trees'
    )

    geoms = list(results)
    gdf = gpd.GeoDataFrame.from_features(geoms)
    gdf.crs = crs  # Assign the Coordinate Reference System of the original image

    # Save to file
    gdf.to_file(filename, driver='GPKG')
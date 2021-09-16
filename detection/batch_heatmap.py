import os
import numpy as np
import torch
import cv2
import argparse
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import CfgNode
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from grad_cam import GradCAM, GradCamPlusPlus

# This code only works for FasterRCNN_C4 architecture
# Can produce heatmap and heatmap++

# constant hardcoded
LAYER_NAME = "roi_heads.res5.2.conv3"
MODEL_ARCHI = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"


def get_args():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")

    parser.add_argument(
        "--clsname_list", 
        nargs="*",
        type=str,
        default=['plane'],
        help="List of class name that your model is trained for, background class is not counted",
    )

    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=['gradcam','gradcam++'],
        default='gradcam',
        help="Visualization method to use",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        help="Path to .pth or .pkl pretrained detectron2 model that you want to visualize",
    )

    parser.add_argument(
        "--img_folder",
        type=str,
        help="A directory contains all images to be detected",
    )
    
    parser.add_argument(
        "--output",
        help="A directory to save output visualizations."
    )

    return parser.parse_args()


def get_model(args) -> torch.nn.Module:
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_ARCHI))
    cfg.MODEL.WEIGHTS = args.pretrained
    # Set the number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(args.clsname_list)
    cfg.MODEL.DEVICE = args.device
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf
    cfg.freeze()
    
    print(cfg)
    model = build_model(cfg)

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    return model, cfg
    

def get_img_input_dict(d2_cfg: CfgNode, img_path: str) -> dict:
    original_image = cv2.imread(img_path)

    height, width = original_image.shape[:2]
    transform_gen = T.ResizeShortestEdge(
        [d2_cfg.INPUT.MIN_SIZE_TEST, d2_cfg.INPUT.MIN_SIZE_TEST], d2_cfg.INPUT.MAX_SIZE_TEST
    )
    image = transform_gen.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(True)

    inputs = {"image": image, "height": height, "width": width}
    return inputs


def combine_mask_to_img(image, mask):
    image = image.copy()
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[..., ::-1]  # gbr to rgb

    image = 0.3 * heatmap + np.float32(image/255)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def save_heatmaps(
    original_img: np.ndarray,
    result_list: list, 
    output_folder: str, 
    img_name: str, 
    clsname_list: list):

    for i, result_dict in enumerate(result_list):
        x1, y1, x2, y2 = result_dict['box']
        conf = result_dict['conf']
        cls_id = result_dict['class_id']
        mask = result_dict['cam']
        cls_name = clsname_list[cls_id]

        crop_area = original_img[y1:y2, x1:x2]
        
        heatmap = combine_mask_to_img(crop_area, mask)
        
        output_path = os.path.join(output_folder,
            '{}_obj_{}_{}_{:.2f}.jpg'.format(img_name, i, cls_name, conf))
        
        # resize to make it larger
        scale = 10
        heatmap = cv2.resize(heatmap, (heatmap.shape[0]*scale, heatmap.shape[1]*10), interpolation = cv2.INTER_CUBIC)

        cv2.imwrite(output_path, heatmap)


# inference on a single image
def gradcam_single_img(args, cfg, gradcam_model, img_path: str, output_folder: str):
    input_dict = get_img_input_dict(cfg, img_path)
    
    result_list = gradcam_model.get_mask_all_detection(input_dict)

    #save image (with all your format)
    img_name = os.path.basename(img_path)
    output_folder = os.path.join(output_folder, args.method)
    os.makedirs(output_folder, exist_ok=True)
    original_img = cv2.imread(img_path)

    save_heatmaps(original_img, result_list, output_folder, img_name, args.clsname_list)


def gradcam(method: str, model):
    if method == 'gradcam':
        return GradCAM(model, LAYER_NAME)
    elif method == 'gradcam++':
        return GradCamPlusPlus(model, LAYER_NAME)


# inference on the folder 
# just call on single image multiple time
def main(args):
    model, cfg = get_model(args)
    gradcam_model = gradcam(args.method, model)

    for img_file in tqdm(os.listdir(args.img_folder)):
        img_path = os.path.join(args.img_folder, img_file)
        gradcam_single_img(args, cfg, gradcam_model, img_path, args.output)


if __name__ == "__main__":
    """
    python detection/batch_heatmap.py --pretrained pretrained_model/your_model.pth \
      --img_folder ./your_folder \
      --output ./your_output_folder \
    """
    arguments = get_args()
    main(arguments)
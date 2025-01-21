# import necessary libraries 
import argparse # for parsing command line arguments 
import time # for time-related operations 
from pathlib import Path # for handling file paths 

import cv2  # OpenCV library for computer vision tasks
import torch  # PyTorch deep learning framework
import torch.backends.cudnn as cudnn  # Backend for CuDNN (CUDA Deep Neural Network library)

# Setting environment variables
import os  # For interacting with the operating system
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # Allowing duplicate libraries for OpenMP

import sys  # For system-specific parameters and functions
sys.path.insert(0, './yolov5') # Setting path to YOLOv5 module without changing base

from yolov5.models.experimental import attempt_load # loading YOLOv5 model 
from yolov5.utils.datasets import LoadStreams, LoadImages # loading image/video data for YOLOv5 model 
from yolov5.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box # general utilities for YOLOv5 model 
from yolov5.utils.plots import colors, plot_one_box # utilities to plot bounding boxes in YOLOv5 
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized # custom functions for visualization

from deep_sort_pytorch.utils.parser import get_config  # Parsing configurations for DeepSort
from deep_sort_pytorch.deep_sort import DeepSort  # DeepSort object tracker

from graphs import bbox_rel, draw_boxes  # Custom functions for visualizing bounding boxes

# Function for detecting objects using YOLOv5 and applying DeepSort for tracking
@torch.no_grad() # Decorator for disabling gradient calculation in PyTorch
def detect(weights='yolov5s.pt',  # model.pt path(s)
           source='yolov5/data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           config_deepsort="deep_sort_pytorch/configs/deep_sort.yaml"  # Deep Sort configuration
           ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # Check if the source is a webcam, a file with a .txt extension, or a URL/stream
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize DeepSort by loading configurations from a specified file
    cfg = get_config()  # Load default DeepSort configurations
    cfg.merge_from_file(config_deepsort)  # Update configurations from the specified file
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,  # Path to the ReID (Re-identification) model checkpoint
        max_dist=cfg.DEEPSORT.MAX_DIST,  # Maximum allowable distance between object embeddings for matching
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,  # Minimum confidence score for object detections
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,  # Maximum overlap allowed for Non-Maximum Suppression (NMS)
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,  # Maximum IoU (Intersection over Union) distance
        max_age=cfg.DEEPSORT.MAX_AGE,  # Maximum age of an object's track
        n_init=cfg.DEEPSORT.N_INIT,  # Number of consecutive frames to activate the tracker
        nn_budget=cfg.DEEPSORT.NN_BUDGET,  # Size of the appearance descriptor distance metric cache
        use_cuda=True  # Specify to use CUDA (GPU) for processing if available
    )

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    if half:
        model.half()  # to FP16

    vid_path, vid_writer = None, None

    # Load data
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=augment)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            if dataset.mode != 'image' and vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous writer
                if vid_cap:
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path += '.mp4'
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                bbox_xywh = []
                confs = []

                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    bbox_xywh.append([x_c, y_c, bbox_w, bbox_h])
                    confs.append([conf.item()])

                outputs = deepsort.update(torch.Tensor(bbox_xywh), torch.Tensor(confs), im0)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    vid_writer.write(im0)

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()

    print(f'Results saved to {save_dir}')
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='ped.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.60, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")

    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'thop'))
    detect(**vars(opt))

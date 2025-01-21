# Object Tracking Using DeepSORT and YOLOv5

This project implements object tracking using **DeepSORT** and **YOLOv5**. The pipeline combines YOLOv5 for object detection and DeepSORT for object tracking to process video files and generate outputs with bounding boxes and tracking IDs.

## Project Structure

├── detect_sort.py                 # Main script for object detection and tracking

├── YOLOv5_DeepSort_Tracking.ipynb # Jupyter notebook for exploration and visualization

├── yolov5s.pt                    # Pretrained YOLOv5 weights

├── requirements.txt              # Python dependencies

├── graphs.py                     # Utility functions for bounding box visualization

├── LICENSE                       # License file (MIT License)

├── pedestrian.mp4                # Input video for demonstration

├── yolov5/                       # YOLOv5 model codebase

├── deep_sort_pytorch/            # DeepSORT implementation

├── runs/                         # Folder containing output videos and results
   
   ├── detect/
         
   -exp/                  # Folder where the processed video is saved

├── __pycache__/                  # Python cache files



## Features

- **Object Detection**: Detect objects in videos using YOLOv5.
- **Object Tracking**: Track detected objects across frames using DeepSORT.
- **Output Generation**: Save processed videos with bounding boxes and tracking IDs in the `runs/detect/exp` directory.
- **Comparison**: View the original video (`pedestrian.mp4`) and the processed video for a clear understanding of the impact of object tracking.


## Installation and Setup

Follow these steps to set up the project and run the object tracking script:

1. **Clone the Repository**:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv .venv
   # Activate the environment:
   # On Windows:
   .venv\Scripts\activate
   # On Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Script**:
   ```bash
   python detect_sort.py --weights yolov5s.pt --img 640 --source pedestrian.mp4 --view-img
   ```


## Input and Output

- **Input Video**:
  - The original video (`pedestrian.mp4`) demonstrates the scene before applying object detection and tracking.

- **Output Video**:
  - The processed video, including bounding boxes and tracking IDs, is saved in the `runs/detect/exp` folder.

  | Original Video (Before Tracking) | Processed Video (After Tracking) |
  |----------------------------------|----------------------------------|
  | ![Original Video](pedestrian.mp4) | ![Processed Video](runs\detect\exp4) |



## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

## Notes

- Ensure the input video (`pedestrian.mp4`) is placed in the root directory.
- The `detect_sort.py` script supports additional configurations like saving detection results as text files, cropping detected objects, and more. Run `python detect_sort.py --help` for all available options.

Feel free to explore and modify the project for your use cases!


import cv2
import torch
from pathlib import Path
from models.yolo import Model
from utils.general import check_requirements, set_logging, non_max_suppression, scale_coords, xyxy2xywh
from utils.google_utils import attempt_download
from utils.torch_utils import select_device
import numpy as np
import random

dependencies = ['torch', 'yaml']

def custom(path_or_model='path/to/model.pt', class_pred=None, autoshape=True):
    """Custom mode

    Arguments (3 options):
        path_or_model (str): 'path/to/model.pt'
        path_or_model (dict): torch.load('path/to/model.pt')
        path_or_model (nn.Module): torch.load('path/to/model.pt')['model']

    Returns:
        pytorch model
    """
    model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    if class_pred is None:
        hub_model.names = model.names  # class names
    else:
        model.names = class_pred
        hub_model.names = class_pred
    if autoshape:
        hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
    return hub_model.to(device)

class_a = ['Pedestrian Crossing', 'Equal-level Intersection', 'No Entry', 'Right Turn Only', 'Intersection', 'Intersection with Uncontrolled Road', 'Dangerous Turn', 'No Left Turn', 'Bus Stop', 'Roundabout', 'No Stopping and No Parking', 'U-Turn Allowed', 'Lane Allocation', 'No Left Turn for Motorcycles', 'Slow Down', 'No Trucks Allowed', 'Narrow Road on the Right', 'No Passenger Cars and Trucks', 'Height Limit', 'No U-Turn', 'No U-Turn and No Right Turn', 'No Cars Allowed', 'Narrow Road on the Left', 'Uneven Road', 'No Two or Three-wheeled Vehicles', 'Customs Checkpoint', 'Motorcycles Only', 'Obstacle on the Road', 'Children Present', 'Trucks and Containers', 'No Motorcycles Allowed', 'Trucks Only', 'Road with Surveillance Camera', 'No Right Turn', 'Series of Dangerous Turns', 'No Containers Allowed', 'No Left or Right Turn', 'No Straight and Right Turn', 'Intersection with T-Junction', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (80km/h)', 'Speed limit (40km/h)', 'Left Turn', 'Low Clearance', 'Other Danger', 'Go Straight', 'No Parking', 'Containers Only', 'No U-Turn for Cars', 'Level Crossing with Barriers', 'X', 'X', 'X', 'X', 'X', 'X', 'X']

model = custom(path_or_model='best.pt', class_pred=class_a)


# Helper function to plot a single bounding box on the image
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break


    #muốn chạy cam thì comment 1 dòng dưới.
    frame=cv2.imread(r'C:\Users\Asus\Downloads\tsy7\images\0001.jpg')
    
    results = model( cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.imshow('YOLO', np.squeeze(results.render()))


    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
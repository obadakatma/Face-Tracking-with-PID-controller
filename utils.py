import cv2
import numpy as np


MARGIN = 10  # pixels
ROW_SIZE = 30  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 0)  # black


def visualize(image,detection_result) -> np.ndarray:
  cord =[0,0]
  count = 0
  
  for detection in detection_result.detections:
    
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    
    # Use the orange color for high visibility.
    cv2.rectangle(image, start_point, end_point, (0, 165, 255), 3)
    cv2.circle(image, (bbox.origin_x + int(bbox.width/2), bbox.origin_y + int(bbox.height/2)),4,(0, 255, 0), -1)
    cord[0]+= bbox.origin_x + int(bbox.width/2)
    cord[1]+= bbox.origin_y + int(bbox.height/2)
    count+=1
    
    # Draw label and score
    category = detection.categories[0]
    category_name = (category.category_name if category.category_name is not None else '')
    
  if count:
      cord[0]//= count
      cord[1]//= count
      cv2.circle(image, tuple(cord),4,(255, 0, 0), -1)
      
  return image,cord

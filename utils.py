# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np


MARGIN = 10  # pixels
ROW_SIZE = 30  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 0)  # black


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualized.
  Returns:
    Image with bounding boxes.
  """
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
    # ~ print((bbox.origin_x + int(bbox.width/2), bbox.origin_y + int(bbox.height/2)))
    # Draw label and score
    category = detection.categories[0]
    category_name = (category.category_name if category.category_name is not
                     None else '')
    # ~ probability = round(category.score, 2)
    # ~ result_text = category_name + ' (' + str(probability) + ')'
    # ~ text_location = (MARGIN + bbox.origin_x,
                     # ~ MARGIN + ROW_SIZE + bbox.origin_y)
    # ~ cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                # ~ FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
  if count:
      cord[0]//= count
      cord[1]//= count
      cv2.circle(image, tuple(cord),4,(255, 0, 0), -1)
  return image,cord

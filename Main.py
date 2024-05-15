import argparse
import sys
import time

import cv2
import mediapipe as mp
import RPi.GPIO as GPIO

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize
from picamera2 import Picamera2 ,Preview

from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory

from Stepper28byj import Stepper28BYJ 
from servoControl import Servo
from test import DynamicPlotWindow
import test 
import threading

import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer



# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None


#PID variables
prevError ,errorSum ,timeDiff , offset , yErrorPrev= 0, 0, 0.1 ,0,0 # offset is optional value for pid output when being zero value
data = 0

def run(model: str, min_detection_confidence: float,
        min_suppression_threshold: float, camera_id: int, width: int,
        height: int) -> None:

  global yErrorPrev, errorSum
  # Start capturing video input from the camera
  picam2 = Picamera2()
  camera_config = picam2.create_preview_configuration({"size": (320, 240)}, raw = picam2.sensor_modes[0])
  picam2.configure(camera_config)
  # ~ picam2.start_preview(Preview.QTGL)
  picam2.start()
  

  controlPins = [31,33,35,37]
  lServoPin = 12

  stepper = Stepper28BYJ(controlPins)
  pigpio_factory = PiGPIOFactory()
  servo = AngularServo(18, pin_factory= pigpio_factory)
  
  stepper.setPinMode(GPIO.BOARD)
  stepper.setSteppingMode(2)
  stepper.init()
  servoDegree = 0
  servo.angle = servoDegree
  time.sleep(2)
  
  
  # Visualization parameters
  row_size = 50  # pixels
  left_margin = 24  # pixels
  text_color = (0, 255, 0)  # green
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  def save_result(result: vision.FaceDetectorResult, unused_output_image: mp.Image,
                  timestamp_ms: int):
      global FPS, COUNTER, START_TIME, DETECTION_RESULT

      # Calculate the FPS
      if COUNTER % fps_avg_frame_count == 0:
          FPS = fps_avg_frame_count / (time.time() - START_TIME)
          START_TIME = time.time()

      DETECTION_RESULT = result
      COUNTER += 1

  # Initialize the face detection model
  base_options = python.BaseOptions(model_asset_path=model)
  options = vision.FaceDetectorOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.LIVE_STREAM,
                                       min_detection_confidence=min_detection_confidence,
                                       min_suppression_threshold=min_suppression_threshold,
                                       result_callback=save_result)
  detector = vision.FaceDetector.create_from_options(options)

 
  # Continuously capture images from the camera and run inference
  while True:
    image = picam2.capture_array()
    image = cv2.flip(image,1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Run face detection using the model.
    detector.detect_async(mp_image, time.time_ns() // 1_000_000)

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(FPS)
    text_location = (left_margin, row_size)
    current_frame = image
    cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                font_size, text_color, font_thickness, cv2.LINE_AA)

    if DETECTION_RESULT:
      if len(DETECTION_RESULT.detections):        
        current_frame , cord = visualize(current_frame, DETECTION_RESULT)
        
        yError, ydir = error(current_frame.shape[0],cord[1])
        if yError!=None:
            
            Kp = 5
            Ki = 0.000001
            Kd = 10
            # Proportional
            P = Kp*ydir*yError
            
            
            #Integral
            errorSum += yError
            I = Ki*ydir*errorSum
          
            
            #Derivative
            D = Kd*ydir*(yError-yErrorPrev)
            yErrorPrev = yError
            
            pidOutput = P + I + D ### PID equation 
            test.data  = pidOutput
            
            #servo
            servoDegree = servoDegree + pidOutput
            
            # Reset line of sight if instructed to look out of bounds            
            if (servoDegree>90 or servoDegree<-90):
                servoDegree = 0
           

            servo.angle = servoDegree
            
            time.sleep(0.00001)


        yError = 0
        ydir = 0
      else:
        servo.angle = servoDegree

    cv2.imshow('face_detection', current_frame)

    if cv2.waitKey(1) & 0XFF == ord(" "):
      break

  detector.close()
  cv2.destroyAllWindows()




def PID(value,setPoint,kP,kI,kD):
  global integral,prevError,timeDiff,offset
  error = setPoint - value
  P = kP * error
  integral = integral + kI*error*(timeDiff)
  D = kD*(error - prevError)/(timeDiff)
  
  pidOutput = P +  integral + D + offset
  
  prevError = error
  return pidOutput  



def Map(value, fromLow, fromHigh, toLow, toHigh):
  return (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow) + toLow



def error(windowMax, x):
    normalised_adjustment = x/windowMax - 0.5
    adjustment_magnitude = abs(round(normalised_adjustment,1))

    if normalised_adjustment>0:
        adjustment_direction = -1
    else:
        adjustment_direction = 1
        
    return adjustment_magnitude, adjustment_direction




def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the face detection model.',
      required=False,
      default='detector.tflite')
  parser.add_argument(
      '--minDetectionConfidence',
      help='The minimum confidence score for the face detection to be '
           'considered successful..',
      required=False,
      type=float,
      default=0.6)
  parser.add_argument(
      '--minSuppressionThreshold',
      help='The minimum non-maximum-suppression threshold for face detection '
           'to be considered overlapped.',
      required=False,
      type=float,
      default=0.6)
  # Finding the camera ID can be very reliant on platform-dependent methods. 
  # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0. 
  # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
  # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=320)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=240)
  args = parser.parse_args()

  run(args.model, args.minDetectionConfidence, args.minSuppressionThreshold,
      int(args.cameraId), args.frameWidth, args.frameHeight)



if __name__ == '__main__':
  main()

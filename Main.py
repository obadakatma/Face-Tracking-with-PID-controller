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

from opencv_multiplot import Plotter


# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None


#PID variables
yErrorSum, yErrorPrev= 0,0
xErrorSum, xErrorPrev= 0,0
cord = 0,0

    
def run(model: str, min_detection_confidence: float,
        min_suppression_threshold: float) -> None:

  global yErrorPrev, yErrorSum, xErrorPrev, xErrorSum, cord
  
  
  # Start capturing video input from the camera
  picam2 = Picamera2()
  camera_config = picam2.create_preview_configuration({"size": (320, 240)}, raw = picam2.sensor_modes[0])
  picam2.configure(camera_config)
  picam2.start()
  
  
  controlPins = [31,33,35,37]
  lServoPin = 12

  stepper = Stepper28BYJ(controlPins)
  pigpio_factory = PiGPIOFactory()
  servo = AngularServo(18, pin_factory= pigpio_factory)
  
  stepper.setPinMode(GPIO.BOARD)
  stepper.setSteppingMode(2)
  stepper.init()
  stepperPrev = 0
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

  plot = Plotter(700, 250, 4)

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
    fps_text = 'FPS={:.1f}'.format(FPS)
    text_location = (left_margin, row_size)
    current_frame = image
    cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                font_size, text_color, font_thickness, cv2.LINE_AA)

    if DETECTION_RESULT:
      if len(DETECTION_RESULT.detections):        
        current_frame , cord = visualize(current_frame, DETECTION_RESULT)
        
        yError, ydir = error(current_frame.shape[0],cord[1])
        xError, xdir = error(current_frame.shape[1],cord[0])
        if yError != None or xError != None:
          
          
            ### Y Term
            yKp = 7.5
            yKi = 0.000001
            yKd = 9
       
       
            # Proportional
            yP = yKp*ydir*yError
            
            #Integral
            yErrorSum += yError
            yI = yKi*ydir*yErrorSum
          
            #Derivative
            yD = yKd*ydir*(yError-yErrorPrev)
            yErrorPrev = yError 
            
            yPidOutput = yP + yI + yD ### PID equation 
            yPidOutput = round(yPidOutput)
            
            ### X Term
            xKp = 10
            xKi = 0.003
            xKd = 8
            
            # Proportional
            xP = xKp*xError
            
            #Integral
            xErrorSum += xError
            xI = xKi*xErrorSum
          
            #Derivative
            xD = xKd*(xError-xErrorPrev)
            xErrorPrev = xError 
            
            xPidOutput = xP + xI + xD ### PID equation 
            xPidOutput = round(xPidOutput)
            xPidOutput = abs(xPidOutput)
            
            #servo
            servoDegree = servoDegree + yPidOutput
            
            # Reset line of sight if instructed to look out of bounds            
            if (servoDegree>90 or servoDegree<-90):
                servoDegree = 0
           

            servo.angle = servoDegree
            
            if xdir == 1:
              stepper.cwStepping(xPidOutput)
              
            elif xdir == -1:
              stepper.ccwStepping(xPidOutput)
              
              
            time.sleep(0.00001)

        
        yError = 0
        ydir = 0
        xError = 0
        xdir = 0
        
      else:
        servo.angle = servoDegree
        
    ### Plotting and Show Frame .....
    plot.multiplot([cord[1],current_frame.shape[0]//2,cord[0],current_frame.shape[1]//2]) 
    cv2.imshow('face_detection', current_frame)

    if cv2.waitKey(1) & 0XFF == ord(" "):
      break

  detector.close()
  cv2.destroyAllWindows()


def error(windowMax, x):
    normalised_adjustment = x/windowMax - 0.5
    adjustment_magnitude = abs(round(normalised_adjustment,1))

    if normalised_adjustment>0:
        adjustment_direction = -1
    else:
        adjustment_direction = 1
        
    return adjustment_magnitude, adjustment_direction

if __name__ == '__main__':
  run("detector.tflite", 0.6, 0.6)

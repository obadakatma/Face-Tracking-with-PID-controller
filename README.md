# Face Tracking with PID controller
Face Tracking device powered by Raspberry pi 3B and controlled using PID controller</br>
This Project for Damascus University, Control Theory Lab

## 3D Design
The design of this project have done on fusion 360 software</br>
You can check the design in ""3D Design"" folder which contains the .f3z fusion file and the .step file.</br>
> **Don't** forget to read the [README.txt](https://github.com/obadakatma/Face-Tracking-with-PID-controller/blob/main/3D%20Design/README.txt) to open the fusion file on your computer.</br>

## 3D Printing
All parts have been printed with PLA+ filament.</br>
**Some settings for printing**</br>
Layer height = 0.15</br>
Wall count = 3 to 4</br>
Top and Buttom layer count = 5</br>
Infill = 25% up to 40% (like the part which holding the Servo and camera printed in 40% infill)</br>
Infill pattern = Cubic</br>

Some parts need support while printing.</br>

## Hardware
This project run on raspberry pi 3B with Raspberry pi camera module V3.</br>
The camera moves up and down using a MG995 Servo and moves right and left using 28BYJ-48 Stepper Motor.</br>
Controlling the servo directly from the Raspberry pi while ULN2003 Driver takes the signal from Raspberry pi and controls the stepper motor</br>
Both servo motor and stepper motor take power from external power supply about 5-7 Volt and 1-2 Amperes.</br>
**Be Sure** that you connect a common ground between power supply and Raspberry pi.</br>

## Software
All the codes written in python programming language.</br>
**Main libraries**</br>
opencv for computer vision and image processing.</br>
mediapipe for face detection.</br>
picamera2 to make an instance to capture the video from the camera.</br>

### How the code works
In General..</br>
The mediapipe Face Detection Model detects the faces and returns its center coordinates, then the calculating the error between the center of the face and our desired setpoint which was the center of camera frame ,**That means there will be two errors one on the X axis and one on the Y axis**, after obtaining the error starts the PID Controller algorithm which takes the error and calculate the Proportional and Integral and Derivative terms, Finaly calculating the PID output by summing the PID Controller terms.</br>
Of course there is two PID Controllers because there is 2DOF in the system.</br>

**Plotting**</br>
Plotting GUI have been done using opencv which only showing drawing commands.</br>
But.... Why the Plotting???</br>
Because it's important to tune the PID Controller constant manually.</br>

### Installing Libraries
```console
pip install requirements.txt
```
### How to run the code
Make sure the camera connected properly to Raspberry pi and all the motors connected.</br>
```console
sudo pigpiod
python Main.py
```
The `sudo pigpiod` is necessary to run the pin factory in gpiozero library.</br>

## The Control Theory
The control algorithm used in this project is PID Control algorithm which provides pretty decent results.</br>
For tuning the controller constants we mainly focused on the plotting and creat a track bar window to choose the values for the constants Kp,Ki,Kd.</br>
This method makes it easier to tune the PID parameters in real-time and visualize the graph in plotting window.</br>

At first make all of Kp,Ki,Kd zero then start tuning the Proportional term which will make an overshoot in the system, then tuning Derivative term which will reduce the over shoot of the system to make it much more reasonable, finaly the system will have a steady state error and by tuning the Integral term it will be so close to zero.</br>

## Project Team
[Obada Katma](https://github.com/obadakatma)</br>
[Omar Jbair](https://github.com/omarjbair)</br>
[Mohammad Al-Salhi](https://github.com/Mohammadalsalhi55)</br>

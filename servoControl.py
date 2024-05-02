import 	RPi.GPIO as GPIO
import time

class Servo:
	def __init__(self,pin):
		self.pin = pin
		self.pwm = 0
		GPIO.setmode(GPIO.BOARD)
		GPIO.setup(self.pin, GPIO.OUT)
		
	def setPwm(self,frequncy,frequncyStart):
		self.pwm = GPIO.PWM(self.pin,frequncy)
		self.pwm.start(frequncyStart)
		
	def writeAngle(self,angle):
		positioner = (angle/18.) + 2
		self.pwm.ChangeDutyCycle(positioner)
	
	def writeDutyCycle(self,dutyCycle):
		self.pwm.ChangeDutyCycle(dutyCycle)
		time.sleep(1)
	
	def clear(self):
		self.pwm.stop()
		

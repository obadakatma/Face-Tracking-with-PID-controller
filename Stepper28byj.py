import RPi.GPIO as GPIO
import time

class Stepper28BYJ:
	__singlePhaseStepping = [
								[1, 0, 0, 0],
								[0, 1, 0, 0],
								[0, 0, 1, 0],
								[0, 0, 0, 1]
							]
	__dualPhaseStepping = [
							[1, 1, 0, 0],
							[0, 1, 1, 0],
							[0, 0, 1, 1],
							[1, 0, 0, 1]
						  ]
	__halfStepping = [
						[1,0,0,0],
						[1,1,0,0],
						[0,1,0,0],
						[0,1,1,0],
						[0,0,1,0],
						[0,0,1,1],
						[0,0,0,1],
						[1,0,0,1]
					 ]
					 
	def __init__(self,controlPinsList):
		self.controlPins = controlPinsList
		self.stepMode = 0
		
	def setPinMode(self,mode):
		GPIO.setmode(mode)
	
	def setSteppingMode(self,mode):
		self.stepMode = mode
		
	def init(self):
		GPIO.setwarnings(False)
		for pin in self.controlPins:
			GPIO.setup(pin,GPIO.OUT)
			GPIO.output(pin,0)
	
	def cwStepping(self,steps):
		for i in range(steps):
			for step in range(4 if self.stepMode == 0 or self.stepMode == 1 else 8):
				for pin in range(4):
					GPIO.output(self.controlPins[pin],self.__singlePhaseStepping[step][pin] if self.stepMode == 0 else self.__dualPhaseStepping[step][pin] if self.stepMode == 1 else self.__halfStepping[step][pin])
				time.sleep(0.001)
				
	def ccwStepping(self,steps):
		for i in range(int(steps)):
			for step in range(4 if self.stepMode == 0 or self.stepMode == 1 else 8):
				for pin in range(4):
					GPIO.output(self.controlPins[pin],self.__singlePhaseStepping[step][len(self.controlPins) - 1 - pin] if self.stepMode == 0 else self.__dualPhaseStepping[step][len(self.controlPins) - 1 - pin] if self.stepMode == 1 else self.__halfStepping[step][len(self.controlPins) - 1 - pin])
				time.sleep(0.001)

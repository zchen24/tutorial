#!/usr/bin/env python3

"""
Show how to use Raspberry Pi PWM mode

RPI only has 2 hardware PWM channels
- BCM12 PWM0
- BCM13 PWM1


p = GPIO.PWM(channel, frequency)
p.start(dc)  # dc = duty cycle
p.ChangeFrequency(freq)
p.ChangeDutyCycle(dc)
p.stop()

"""

from RPi import GPIO
import time


PIN_LED = 12

# use BCM pin#
GPIO.setmode(GPIO.BCM)

# setup
GPIO.setup(PIN_LED, GPIO.OUT)
p = GPIO.PWM(PIN_LED, 500)

p.start(0)

for i in range(100):
    p.ChangeDutyCycle(i)
    time.sleep(0.1)

for i in range(100):
    p.ChangeDutyCycle(100-i)
    time.sleep(0.1)

p.stop()

# cleanup
GPIO.cleanup()

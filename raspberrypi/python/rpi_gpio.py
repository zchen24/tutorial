#!/usr/bin/env python3

"""
Show how to use Raspberry Pi GPIO
"""

from RPi import GPIO
import time


PIN_BTN = 11
PIN_LED = 12


def callback_gpio(pin):
    print('callback_gpio, pin# {}'.format(pin))


# use BCM pin#
GPIO.setmode(GPIO.BCM)

# setup
GPIO.setup(PIN_LED, GPIO.OUT)
GPIO.setup(PIN_BTN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ----------------------
#  polling
# ----------------------
for _ in range(100):
    val = GPIO.input(PIN_BTN)   # read button
    GPIO.output(PIN_LED, val)   # output led
    time.sleep(0.1)

# ----------------------
#  event callback
# ----------------------
GPIO.add_event_detect(PIN_BTN, GPIO.RISING)
GPIO.add_event_callback(PIN_BTN, callback_gpio)
time.sleep(10)  # wait 10 seconds

# cleanup
GPIO.cleanup()

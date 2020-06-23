#!/usr/bin/env python3

import time

pin_skip = False

try:
    import RPi.GPIO as GPIO
    led_pin_a = 12
    led_pin_b = 16
    print('load rpi gpio')
except:
    try:
        import Jetson.GPIO as GPIO
        led_pin_a = 12
        led_pin_b = 16
        print('load jetson gpio')
    except:
        pin_skip = True
        print('no load gpio')


class Game:
    def __init__(self):
        print('hello - gpio test')

    def pin_setup(self):
        if pin_skip: return
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(led_pin_a, GPIO.OUT)
        GPIO.setup(led_pin_b, GPIO.OUT)

    def pin_a_on(self):
        if pin_skip: return
        GPIO.output(led_pin_a, GPIO.HIGH)
        GPIO.output(led_pin_b, GPIO.LOW)

    def pin_a_off(self):
        if pin_skip: return
        GPIO.output(led_pin_a, GPIO.LOW)
        GPIO.output(led_pin_b, GPIO.HIGH)

if __name__ == '__main__':

    g = Game()
    g.pin_setup()
    while True:
        print('light on')
        g.pin_a_on()
        time.sleep(1)
        print('light off')
        g.pin_a_off()
        time.sleep(1)
## Indicator Lights

The `game.py` file has been edited for two indicator lights that are to be connected to the GPIO pins on the Raspberry Pi.
Explanation of how the GPIO pins work is beyond the scope of this document.
The GPIO library automatically loads when the code is run on the Pi.
It is not loaded on a x86_64 machine.
The pins used are listed below.

* 06 - GND
* 12 - LED A -- GREEN
* 14 - GND
* 16 - LED B -- RED


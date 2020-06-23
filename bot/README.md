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

Further, we are going to try to install a shutdown button on the following gpio pins.

* 05 - BUTTON +
* 20 - GND

The line to be added to the `/etc/rc.local` file is as follows:

```
sudo python /home/pi/workspace/awesome-chatbot/shutdown.py &
```

## Alternate Wiring
The alternate wiring is simply so that the `button+` and `button-` are on wire `05` and `06` respectively.

* 14 - GND
* 12 - LED A -- GREEN
* 20 - GND
* 16 - LED B -- RED

Further, we are going to try to install a shutdown button on the following gpio pins.

* 05 - BUTTON +
* 06 - GND

import random
from pynput.keyboard import Key, Controller
import time

keyboard = Controller()
perc = 10

while True:
    if (random.random() < perc/100):
        keyboard.press(Key.home)
        keyboard.release(Key.home)
        
    time.sleep(1)
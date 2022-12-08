import time

from pynput.keyboard import Key, Controller


if __name__ == '__main__':

    time.sleep(10)

    code = 'x = 1 // 2 + b * c % 10 - 6 % 3'

    keyboard = Controller()

    while True:
        for c in code:
            time.sleep(10)
            keyboard.type(c)
        for c in code:
            time.sleep(10)
            keyboard.press(Key.backspace)
            keyboard.release(Key.backspace)
        time.sleep(10)
import os
import serial

time_values = [0.0, 2.5, 2.5, 4.5, 5.0, 10.0, 7.5, 9.5]


if os.name == 'nt':  # sys.platform == 'win32':
    from serial.tools.list_ports_windows import comports
elif os.name == 'posix':
    from serial.tools.list_ports_posix import comports


def get_arduino_port():
    ports = [port for (port, desc, hwid) in comports()]
    assert(len(ports) == 1)
    return ports[0]


class BoardCameraTrigger(object):
    """Class for controlling the Cerebro circuit board camera-trigger.
    """
    def __init__(self, port=None):

        if port is None:
            port = get_arduino_port()

        self.set_port(port)

    def set_port(self, port):
        self.port = port
        self.serial = serial.Serial(port, 115200, timeout=10)

    def initialize(self):
        self.set_defaults()
        self.enable()

    def set_defaults(self):
        command = 's 4000 200 126 126 126 126 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 126 126 126 126 126 126 126 126 126 126 0 0 0 0 0 0 0 0 0 0 255 2500 5000 5050 7500 7500 10000 1000'
        self.write_serial(command)

    def enable(self):
        command = 'e'
        self.write_serial(command)

    def start_triggering(self):
        print("start_triggering")
        self.write_serial("r")

    def stop_triggering(self):
        print("stop_triggering")
        self.write_serial("s")

    def write_serial(self, command):
        command = command + "\n"
        self.serial.write(command.encode())

    def set_time(self, idx, ms):
        # TODO recompute the command string
        self.set_params()

    def set_duration(self, ms):
        # TODO recompute the command string
        self.set_params()

    def set_mode(self, mode):
        # TODO recompute the command string
        self.set_params()



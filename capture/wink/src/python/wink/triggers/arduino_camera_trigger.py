import os
import platform
import serial
import time

time_values = [0.0, 2.5, 2.5, 4.5, 5.0, 10.0, 7.5, 9.5]


if os.name == 'nt':  # sys.platform == 'win32':
    from serial.tools.list_ports_windows import comports
elif os.name == 'posix':
    from serial.tools.list_ports_posix import comports


def get_arduino_port():
    ports = [port for (port, desc, hwid) in comports()]
    system = platform.system()
    if system == 'Linux':
        assert(len(ports) == 1)
        return ports[0]
    elif system == 'Darwin':
        ports = [port for port in ports if 'usbmodem' in port]
        assert(len(ports) == 1)
        return ports[0]
    else:
        raise Exception('get_arduino_port() not implemented for your OS')


class ArduinoCameraTrigger(object):
    """Class for controlling an arduino-based camera-trigger.
    """
    def __init__(self, port=None):

        if port is None:
            port = get_arduino_port()

        self.set_port(port)

    def set_port(self, port):
        self.port = port
        self.serial = serial.Serial(port, 9600, timeout=10)
        time.sleep(3)

    def init_times(self):
        for i in range(len(time_values)):
            self.set_time(i, time_values[i])

    def start_triggering(self):
        print("start_triggering")
        self.write_serial("start")

    def stop_triggering(self):
        print("stop_triggering")
        self.write_serial("stop")

    def write_serial(self, command):
        command = command + "\n"
        self.serial.write(command.encode())
        self.serial.flush()

    def set_time(self, idx, ms):
        command = 'settime:{idx}:{ms}'.format(idx=idx, ms=ms)
        self.write_serial(command)

    def set_duration(self, ms):
        command = 'setduration:{ms}'.format(ms=ms)
        self.write_serial(command)

    def close(self):
        self.serial.close()

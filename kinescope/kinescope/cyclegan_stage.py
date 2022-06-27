import threading
from time import sleep
from queue import Queue

from cyclegan import CycleGanNetwork
from realtime_pipeline import Stage

class CycleGanStage(Stage):

    def __init__(self):
        Stage.__init__(self)
        self.q = Queue(maxsize=1)
        self.net = CycleGanNetwork()
        self.create_worker_thread()

    def create_worker_thread(self):
        t = threading.Thread(target=self.worker, args=())
        t.daemon = True
        t.start()
        self.worker = t

    def worker(self):
        while True:
            image = self.q.get()
            if self.net.can_process(image):
                output_data = self.net(image)
                self.emit(output_data)

    def consume(self, input_data):
        if not self.q.full():
            self.q.put(input_data)

    def load(self, path):
        self.net.load(path)

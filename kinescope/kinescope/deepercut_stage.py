import threading
from time import sleep
import os
from queue import Queue
import sys

deepercut_path = os.path.join(os.path.dirname(__file__),
                              "3rd-party/pose-tensorflow/")
sys.path.append(deepercut_path)

from deepercut import DeeperCutNetwork
from realtime_pipeline import Stage
from util.visualize import _npcircle
import matplotlib.pyplot as plt


class DeeperCutStage(Stage):

    def __init__(self):
        Stage.__init__(self)
        self.q = Queue(maxsize=1)
        self.net = DeeperCutNetwork()
        self.colors = [(r*256, g*256, b*256) for (r, g, b) in plt.get_cmap('tab10').colors]
        self.conf_threshold = 0.9
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
                scmap, predictions = self.net(image)
#                output = np.repeat(scmap, 3, axis=2)
#                output = output * 255
                output = image.copy()
                n_joints = predictions.shape[0]
                print('n_joints = {}'.format(n_joints))
                joints = []
                for i in range(n_joints):
                    cx, cy, conf = predictions[i, 0:3]
                    cx_normalized = cx / output.shape[1]
                    cy_normalized = cy / output.shape[0]
                    joints.append((cx_normalized, cy_normalized, conf))
                    if conf > self.conf_threshold:
                        try:
                            _npcircle(output, cx, cy, 10, self.colors[i],
                                      transparency=0.5)
                        except Exception:
                            print('failed to draw circle')
                output_pair = (output, joints)
                self.emit(output_pair)

    def set_conf_threshold(self, threshold):
        self.conf_threshold = threshold

    def consume(self, input_data):
        if not self.q.full():
            self.q.put(input_data)

    def load(self, path):
        self.net.load(path)

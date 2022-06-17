from collections import deque
from collections import namedtuple
import os
from queue import Queue
import threading


class Stage(object):
    """Pipeline stage base class."""

    def __init__(self):
        self.output_stages = []

    def connect(self, output_stage):
        self.output_stages.append(output_stage)

    def emit(self, output_data):
        for output_stage in self.output_stages:
            output_stage.consume(output_data)

    def consume(self, input_data):
        raise NotImplementedError()


class StatelessStage(Stage):
    """A pipeline stage that applies a function to input data and, optionally,
    passes data to its output stages by calling the emit callback.

    The function has the spec:
    (input_data, emit) -> None
    """

    def __init__(self, function):
        super(StatelessStage, self).__init__()
        self.function = function

    def consume(self, input_data):
        self.function(input_data, self.emit)


class StatefulStage(Stage):
    """A pipeline stage that applies a function to input data and stored state,
    and, optionally, passes data to its output stages via the emit callback or
    changes state via the set_state callback.

    The function has the spec: (input_data, state, set_state, emit) -> None
    """

    def __init__(self, function, initial_state):
        super(StatefulStage, self).__init__()
        self.function = function
        self.state = initial_state

    def consume(self, input_data):
        self.function(input_data, self.state, self.set_state, self.emit)

    def set_state(self, new_state):
        self.state = new_state


# Pairer stage

class Pairer(Stage):
    """A pipeline stage that groups input data into pairs before emitting it."""

    def __init__(self):
        super(Pairer, self).__init__()
        self.data = None
        self.is_empty = True

    def consume(self, input_data):
        if self.is_empty:
            self.data = input_data
            self.is_empty = False
        else:
            pair = (self.data, input_data)
            self.data = None
            self.is_empty = True
            self.emit(pair)

    def clear(self):
        self.data = None
        self.is_empty = True

# Indexer stage

class Indexer(StatelessStage):
    """Pipeline stage which extracts element a given index."""
    def __init__(self, index):
        super(Indexer, self).__init__(lambda tup, emit: emit(tup[index]))

# Skipper stage (used to skip frames)

class Skipper(Stage):
    """Pipeline stage that skips n datums before emitting one more."""

    def __init__(self, num_frames):
        super(Skipper, self).__init__()

        assert(type(num_frames) is int)
        assert(num_frames >= 0)

        self.num_frames = num_frames
        self.skip_count = 0

    def consume(self, input_data):
        if self.skip_count >= self.num_frames:
            self.skip_count = 0
            self.emit(input_data)
        else:
            self.skip_count += 1

    def set_num_frames(self, n):
        self.num_frames = n

    def clear(self):
        self.skip_count = 0

# Buffer stage (used to buffer back frames)

class Buffer(Stage):
    """A pipeline stage which either: a) pushes input data into a queue (if
    buffering is True), or b) emits it. """

    def __init__(self, max_size, buffering):
        super(Buffer, self).__init__()
        self.buffering = True
        self.max_size = max_size
        self.queue = deque(maxlen=self.max_size)

    def set_max_size(self, new_size):
        self.max_size = new_size
        self.queue = deque(maxlen=self.max_size)

    def consume(self, input_data):
        if self.buffering:
            if len(self.queue) == self.max_size:
                 self.queue.pop()
            self.queue.appendleft(input_data)
        else:
            self.emit(input_data)

    def flush(self):
        """Sequentially remove and emit all data in the queue."""
        while len(self.queue) > 0:
            try:
                element = self.queue.pop()
            except:
                continue
            self.emit(element)

    def clear(self):
        self.queue.clear()

# Counter stage (used to stop capturing after N frames)

class Counter(Stage):
    """Pipeline stage which calls a callback after n datums."""

    def __init__(self, n, callback):
        super(Counter, self).__init__()
        self.n = n
        self.count = 0
        self.callback = callback

    def consume(self, input_data):
        if self.count == self.n:
            self.callback()
        else:
            self.count += 1
            self.emit(input_data)

# Batcher stage
#   input:  datum
#   output: (batch, index, datum)

class Batcher(Stage):
    """Pipeline stage which turns datums into (batch, index, datum) tuples.

    batch - int which denotes batch number
    index - int which denotes position within a batch
    """

    def __init__(self):
        super(Batcher, self).__init__()
        self.batch = 0
        self.index = 0

    def consume(self, input_data):
        output_data = (self.batch, self.index, input_data)
        self.index += 1
        self.emit(output_data)

    def new_batch(self):
        self.batch += 1
        self.index = 0

# Splitter
#   input:    datum
#   output_0: (datum[0], datum[1], ... , datum[index][0], ... , datum[n-1])
#   output_1: (datum[0], datum[1], ... , datum[index][1], ... , datum[n-1])
#   output_k: (datum[0], datum[1], ... , datum[index][k], ... , datum[n-1])

class Splitter(Stage):
    """Pipeline stage which splits the iterable located at a given index in
    datum, emitting different values to different output stages. Values at all
    other indices are copied."""

    def __init__(self, index):
        super(Splitter, self).__init__()
        self.index = index

    def consume(self, input_data):
        tup_to_split = input_data[self.index]
        assert(len(self.output_stages) == len(tup_to_split))

        output_values = []
        for i in range(len(tup_to_split)):
            value = tup_to_split[i]
            output_data = replace_at_index(input_data, self.index, value)
            output_values.append(output_data)
        split_emit(output_values)

    def split_emit(self, values):
        for value, output_stage in zip(values, self.output_stages):
            output_stage.consume(value)

    def emit(self):
        raise Exception('Split pipeline stages should only use "split_emit()".')

def replace_at_index(tup, ix, val):
    return tup[:ix] + (val,) + tup[ix+1:]

# FrameWriter
#   input: (batch, index, frame)
#   output: None

def FrameWriter(Stage):
    """Pipeline stage which consumes (batch, index, frame) and
    writes image and metadata to disk."""

    def __init__(self, data_dir, metadata_fname='metadata.txt', n_workers=1):
        super(FrameWriter, self).__init__()
        self.n_workers = n_workers
        self.workers = []
        self.q = Queue()
        self.set_data_dir(data_dir)
        self.init_workers()

    def consume(self, input_data):
        (batch, index, frame) = input_data

        metadata = get_metadata(batch, index, frame)
        self.save_metadata(self.metadata_fname, metadata)

        fpath = get_filepath(batch, index, frame)
        self.async_save_image(fpath, frame.image)

    def save_metadata(self, fpath, metadata):
        print('save_metadata not implemented')

    def async_save_image(self, fpath, image):
        self.q.put((fpath, image), False)

    def set_data_dir(self, data_dir, metadata_fname='metadata.txt'):
        self.data_dir = data_dir
        self.metadata_fname = os.path.join(self.data_dir, metadata_fname)

    def init_workers(self):
        for _ in range(self.n_workers):
            t = threading.Thread(target=self.worker, args=())
            self.workers.append(t)
            t.daemon = True
            t.start()

    def worker(self):
        while True:
            (fpath, image) = self.q.get()
            save_image(fpath, image)
            self.q.task_done()

def get_filepath(batch, index, frame):
    print('get_filepath not implemented')

def get_metadata(batch, index, frame):
    print('get_metadata not implemented')

def save_image(fpath, image):
    print('save_image not implemented')

def create_output_dirs(self, base_dir):
    self.uv_output_dir = base_dir + '/uv'
    self.red_output_dir = base_dir + '/red'
    try:
        os.makedirs(self.uv_output_dir)
    except:
        pass
    try:
        os.makedirs(self.red_output_dir)
    except:
        pass


class FrameGrouper(Stage):

    def __init__(self, num_cams, interleaved_streams=1):
        super().__init__()
        self.num_cams = num_cams
        self.interleaved_streams = interleaved_streams
        self.num_streams = num_cams * interleaved_streams
        self.data = {}
        self.cam_current_stream = [0 for _ in range(self.num_cams)]

    def consume(self, frame):
        cam = frame.camera_context
        stream = self.cam_current_stream[cam]
        label = (cam, stream)

        self.cam_current_stream[cam] = (stream + 1) % self.interleaved_streams

        self.data[label] = frame

        if self.data_ready():
            tup = self.get_data_tup()
            self.emit(tup)
            self.data = {}

    def get_stream_labels(self):
        labels = []
        for cam in range(self.num_cams):
            for stream in range(self.interleaved_streams):
                labels.append((cam, stream))
        return labels

    def data_ready(self):
        return (len(self.data) == self.num_streams)

    def get_data_tup(self):
        tup = tuple(self.data[label] for label in self.get_stream_labels())
        return tup


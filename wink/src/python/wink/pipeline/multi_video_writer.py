import cv2

from wink.pipeline import Stage
from wink.writers import ThreadedWriter


class MultiVideoWriterStage(Stage):

    def __init__(self, video_paths, quality=10, fps=30, debayer=True, nowrite=False, timestamp_file=None):
        super().__init__()
        self.video_paths = video_paths
        self.quality = quality
        self.fps = fps
        self.debayer = debayer
        self.timestamp_file = timestamp_file
        self.nowrite = nowrite
        
        self.writers = []

        if not self.nowrite:
            for video_path in self.video_paths:
                writer = ThreadedWriter(video_path, quality=quality, fps=fps, output_params=['-threads', '1'])
                self.writers.append(writer)

    def consume(self, frame_tup):

        if self.timestamp_file is not None:
            with open(self.timestamp_file, 'a') as f:
                line = "\t".join([str(frame.timestamp) for frame in frame_tup])
                f.write(line + '\n')

        if not self.nowrite:
            assert(len(frame_tup) == len(self.writers))

            for (frame, writer) in zip(frame_tup, self.writers):

                image = frame.image

                if self.debayer and not frame.is_debayered:
                    image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2BGR)

                writer.append_data(image)
        
    def is_done(self):
        return all([writer.is_done() for writer in self.writers])

    def close(self):
        for writer in self.writers:
            writer.close()


from wink.pipeline import Stage

class MultiCameraStage(Stage):

    def __init__(self, multi_camera, split_output_streams=True):
        super().__init__()
        self.split_output_streams = split_output_streams

        self.multi_camera = multi_camera
        self.multi_camera.register_handler(self.handler)

    def handler(self, frame):
        if self.split_output_streams:
            output_stage = self.output_stages[frame.camera_context]
            output_stage.consume(frame)
        else:
            for output_stage in self.output_stages:
                output_stage.consume(frame)

    def capture(self, *args, **kwargs):
        self.multi_camera.capture(*args, **kwargs)

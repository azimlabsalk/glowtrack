from wink.pipeline import Stage

class CameraStage(Stage):

    def __init__(self, camera):
        super().__init__()

        self.camera = camera
        self.camera.register_handler(self.handler)

    def handler(self, grab_result):
        # COPY grab_result ???
        self.emit(grab_result)

    def start_grabbing(self):
        self.camera.start_grabbing()

    def stop_grabbing(self):
        self.camera.stop_grabbing()

    def start_triggering(self):
        self.camera.start_triggering()

    def stop_triggering(self):
        self.camera.stop_triggering()

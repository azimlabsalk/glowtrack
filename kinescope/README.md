# Kinescope

Kinescope is a custom GUI for real-time viewing of 1) strobed fluorescence video, and 2) motion capture model output. These can be helpful for optimizing the GlowTrack video capture rig.

In the GlowTrack paper, Kinescope was the software used to implement the real-time feedback experiment shown in Supp. Fig. 4. For details, see `kinescope/realtime_feedback_widget.py`.

Kinescope can also be used to experiment with different timing configurations for triggering lights and cameras, as shown in Fig. 2b and Fig 2c. Kinescope implements a GUI to allow real-time adjustment of these parameters.

# Dependencies

*Operating system*: Kinescope was written for Linux and has been tested on Ubuntu 16.04.

*Software dependencies*: For capturing video from USB3 cameras, Basler Pylon 5 (proprietary API for Basler cameras) is used, along with its python bindings. For the GUI, PyQt5 is used.

*Hardware dependencies*: A Basler USB3 camera. Kinescope has been tested with the Ace2 series of cameras.

## Usage

To use Kinescope, install the dependencies and run:

```
$ cd kinescope/kinescope
$ python kinescope.py
```

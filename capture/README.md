# Video capture rig

Hardware & software for capturing strobed fluorescent video for the GlowTrack system.

## Components

The main components of the video capture rig are:

- A [geodesic dome](https://github.com/azimlabsalk/glowtrack/blob/main/capture/dome.md) for holding lights and cameras
- An Arduino microcontroller (Arduino Due running [this code](https://github.com/azimlabsalk/glowtrack/blob/main/capture/arduino/MultiStrobe/MultiStrobe.ino)) for triggering the cameras and lights
- [Circuitry](https://github.com/azimlabsalk/glowtrack/blob/main/capture/electronics/schematic.pdf) for controlling lights and cameras
- [Video capture software](https://github.com/azimlabsalk/glowtrack/tree/main/capture/wink) for capturing video from standard machine vision cameras

## Dome & circuitry

To reproduce our hardware setup, we recommend examining our description of the [geodesic dome](https://github.com/azimlabsalk/glowtrack/blob/main/capture/dome.md) and the schematic of the [circuitry](https://github.com/azimlabsalk/glowtrack/blob/main/capture/electronics/schematic.pdf).

## Arduino

1. Install the Arduino IDE
2. From the Arduino IDE, open `arduino/MultiStrobe/MultiStrobe.ino`
3. Attach your Arduino to your computer via USB and upload the MultiStrobe code to it
4. Unplug the Arduino from your computer, and connect your lights and cameras to it (see [our schematic](https://github.com/azimlabsalk/glowtrack/blob/main/capture/electronics/schematic.pdf) for one example)
5. Attach the Arduino to your computer again, and initiate image capture by running a video capture script (see our example code)

## Video capture

For video capture, we wrote code that uses the Basler Pylon API. First, we wrote a python package called [wink](https://github.com/azimlabsalk/glowtrack/blob/main/capture/wink/src/python/wink). This code was performant enough to capture from up to four USB3 cameras. In order to capture from all eight cameras, we also wrote a [C++ script](https://github.com/azimlabsalk/glowtrack/blob/main/capture/wink/src/c++/Grab_MultipleCameras.cpp). Depending on your specific needs, one or the other may be more convenient, so we provide both. Note that the C++ code depends on the python package for sending serial commands to the Arduino, but this dependency could easily be removed. 

To install the python package, follow the instructions in the [wink README](https://github.com/azimlabsalk/glowtrack/blob/main/capture/wink/README.md).

To build the C++ script:

1. Install the OpenCV and Pylon 5
2. Change your working directory: `cd wink/src/c++`
3. To build the script, run `mkdir build && make build/Grab_MultipleCameras`. 

To capture video clips:

1. Make sure your Arduino is loaded with the code, connected to your cameras, and connected to your capture computer via USB. 
2. To capture video clips with full fluorescence strobing, run `./capture.sh [DATA_DIR] [EVERY_NTH] [BUFFER_SIZE] [BATCH_SIZE]`. See comments below for a description of the parameters. Pressing enter once will start triggering the cameras, but not recording. You can now choose to start recording a clip (by pressing enter) or quit (by pressing 'q' + enter). While recording a clip, you can stop recording any time by pressing enter.
3. To deinterleave the visible and UV video frames, run `./deinterleave.sh [DATA_DIR]`.

Description of parameters:

```
EVERY_NTH - Along with BATCH_SIZE, helps determine how many image pairs (each containing one visible and one UV image) to drop between capture batches. A value of 1 indicates that a batch is initiated at every frame pair, while a value of 50 means that a batch is initiated on every 50th image pair. EVERY_NTH must be at least 1 and greater than or equal to BATCH_SIZE.

BUFFER_SIZE - How many image pairs to store in a temporary buffer while not actively recording a clip. When you start recording a clip, the buffer is included at the beginning of the recording, effectively allowing you to record into the past. This is useful if you are capturing brief events like a mouse reaching for a food pellet.

BATCH_SIZE - How many image pairs to capture in a batch. BATCH_SIZE must be at least 1 and less than or equal to EVERY_NTH.

Example 1: ./capture.sh [DATA_DIR] 1 100 1

  This means: 1. capture every image pair, and 2. buffer the most recent 100 image pairs (which are immediately saved when a clip is recorded).

Example 2: ./capture.sh [DATA_DIR] 50 0 2

  This means: 1. repeatedly capture 2 image pairs then drop 48 image pairs, and 2. do not to buffer any image pairs.

```


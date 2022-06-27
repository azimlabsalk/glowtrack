# GlowTrack

A system for training motion capture models via fluorescence labeling.

## Components

The main components of the system are:

- A video capture rig (`capture/`) for capturing strobed fluorescence video.
- A software package, Yogi (`yogi/`), for processing strobed flourescence video, creating motion capture models from it, and evaluating those models.
- A GUI, Kinescope (`kinescope/`), for realtime viewing of 1) strobed fluorescence video, and 2) motion capture model output. These can be helpful for optimizing the video capture rig.

## Demo

The main software demo lives in `yogi/`. It demonstrates a neural network trained on fluorescence-derived labels.

## Figures

This repository contains all code used to produce the figures in the GlowTrack paper. 

The images of mice and the human hand in Fig. 2, Fig. 3ef, Fig. 5, Fig. 6, Supp. Fig. 1, Supp. Fig 2., Supp. Fig. 5b, Supp. Fig. 6, and Supp. Fig. 7 were generated with the strobed-fluorescence video capture rig (`capture/`).

The code and designs for the video capture rig shown in Fig. 3 are located in `capture/`.

The precision-recall curves and error quartile plots in Fig. 3gh, Fig. 4e, Fig. 6eik, Supp. Fig. 3, and Supp. Fig. 4b were generated with the scripts in `yogi/scripts/plots`. These scripts rely on evaluation code in `yogi/yogi/evaluation.py`.

The custom image augmentation procedure depicted in Supp. Fig. 1 is implemented in `yogi/yogi/image_aug.py`.

The parallel labeling approach with SIFT features depicted in Fig. 5, Fig. 6, Supp. Fig. 5, Supp. Fig. 6, and Supp. Fig. 7, is implemented in scripts in `yogi/scripts/speckle` and utility code in `yogi/yogi/matching.py`.

The template selection algorithm shown in Supp. Fig. 5 is implemented in `yogi/dependencies/VocabTree2` (modified version of the existing `VocabTree2` code by Noah Snavely). In particular, see `yogi/dependencies/VocabTree2/VocabCover/VocabCover.cpp`.

## Disclaimer

The code and designs in this repository are intended to serve as a reference for others who want to implement a similar system. They have not been extensively tested. We found them to be sufficient to serve as a proof-of-principle for research purposes. However, for applications with more stringent requirements, additional testing and validation would be necessary.

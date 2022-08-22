# wink

A library for camera control and image capture.

# Installation

### Linux / Windows

First, install Pylon and its Python wrapper:

 * Install [Pylon 5.2](https://www.baslerweb.com/pylon).
 * Install [Pylon Supplementary Package for MPEG-4](https://www.baslerweb.com/en/downloads/software-downloads/pylon-supplementary-package-for-mpeg4-linux-x86-64-bit/).
 * Download a binary wheel from the [releases](https://github.com/Basler/pypylon/releases) page. 
 * Install the wheel using ```pip3 install <your downloaded wheel>.whl```

Then, install wink:  

```bash
git clone https://dbutler@bitbucket.org/dbutler/wink.git
cd wink
pip install -e .
```

### Mac

First, install [pylon](https://www.baslerweb.com/pylon).  

Then, install its Python wrapper from source. Version 1.4.0 seems to work right out of the box, whereas later versions are finicky.  
 
```bash
git clone https://github.com/basler/pypylon.git
cd pypylon
git checkout 1.4.0
pip install .
```

Then, install wink:  

```bash
git clone https://dbutler@bitbucket.org/dbutler/wink.git
cd wink
pip install -e .
```


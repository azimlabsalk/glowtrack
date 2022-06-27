# Yogi

Yogi is the main software package used to process data for the GlowTrack project.

## System Requirements

*Operating system*: Yogi itself runs on Linux (`Ubuntu 16.04.7 LTS`), but the dependencies are somewhat complex. Therefore we recommend running Yogi inside a Docker container (we have included a Dockerfile). We have tested the included Dockerfile on the following host operating systems: `Debian GNU/Linux 9` and `Ubuntu 20.04 LTS`.

*Software dependencies*: A version of the [Docker Desktop](https://docs.docker.com/desktop/linux/install/ubuntu/) host that supports NVIDIA GPUs and Docker Compose. For GPU support, you may need to install the [NVIDIA Toolkit for Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). For running the convenience commands included in our Makefile, the commandline tool `make` is also necessary, but you can manually copy and run those commands from the Makefile into a terminal if you prefer.

*Hardware dependencies*: An NVIDIA GPU compatible with `CUDA 9.0` and `cudnn 7`. We have tested on the following GPUs: GeForce GTX 1070, Tesla P100, Tesla K40c. The minimum GPU memory we have tested on is 8GB.

## Installation

Clone or download this repository. For example:

```
$ cd path/to/install
$ git clone https://github.com/azimlabsalk/glowtrack.git
```

Then, make the Docker image.

```
$ cd glowtrack/yogi
$ make build-py
```

*Note*: to build the Docker image, this step downloads several GBs of software and can take 20-30 minutes. The final docker image is about 4.7GB.

To test that the Docker image was built successfully, run:

```
$ sudo docker image ls
```

There should be an image called `yogi-py`.

To make sure Docker uses the correct permissions, you should tell Docker your UID. Run:

```
$ echo $UID
```

Then, edit `yogi/.env` and set the `UID` variable to the number returned by the above command.

Now, you need to initialize the Postgres database that keeps track of the images, videos, neural networks, and labels.

To start the Postgres Docker service, inside `glowtrack/yogi`:

```
$ make db
```

To check that the database service was started correctly, run `sudo docker compose ps`.

Now, to initialize the database, run:

```
$ make yogi-shell
root@198e92f599c5:/code# alembic upgrade head
```

The first line brings up a shell inside a Docker container, and the second line runs the command to initialize the database.

To test that the database was initialized correctly, run:

```
root@198e92f599c5:/code# yogi shell
>>> session.query(Model).all()
[]
```

The database does not contain any Model objects, so the query should return the empty list `[]`.

To shut down the database service, run:

```
$ make down
```

Yogi is now installed and ready to run the demo!

## Demo

Before you running the demo, you need to install Yogi (see above).

The demo takes about 1-2 hours depending on the speed of your GPU. You can make it run faster by removing some of the mouse demo videos from `data/demo-videos`.

To run the demo:

```
$ cd /your/path/to/glowtrack
$ wget https://cnl.salk.edu/~dbutler/glowtrack/yogi-all-demo-data.zip
$ unzip yogi-all-demo-data.zip
$ make db
$ make yogi-shell
root@198e92f599c5:/code# ./scripts/run_demo.sh
```

You will need to type "y" twice at the very beginning of the script to confirm that you want to create new clip sets; after that, it will run to completion without any interaction.

The demo script runs the main models from the paper on the videos in `data/demo-videos`. You can add extra videos to that directory if desired.

The expected outputs are videos located in `data/output`. Each video should have small dots representing the predicted landmark positions, and text representing the prediction confidence. 

## Usage

The main entrypoint to Yogi is the `yogi` command. Run `make yogi-shell` (to start a Docker container) and then run `yogi` with no arguments to see usage.

The trained models in `data/yogi-dir/models` use the Deepercut architecture, and are therefore compatible with DeepLabCut (DLC). Note, however, that running these models with DLC will not include scale optimization, so image scale must be adjusted appropriately.

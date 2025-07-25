# RL planner
This is a prototype of RL planner.

## Requirements

This project has been developed and tested with the following dependencies:

* `numpy==1.17.0`
* `torch==1.9.0`
* `gym==0.14.0`
* `transforms3d==0.4.1`
* `tensorboardX==2.6.2.2`
* `protobuf==3.20.1`
* Python 3.7

**Note:** While the project was developed with these specific versions, it should be compatible with newer versions of the listed libraries and Python.

## Installation

To set up your environment, we recommend using a virtual environment.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows

# Install the required packages
pip install numpy==1.17.0 torch==1.9.0 gym==0.14.0 transforms3d==0.4.1 tensorboardX==2.6.2.2 protobuf==3.20.1

## Training

To run training, type "python3 main.py --container_size 4 4 4 --item-size-range 2 2 2 2 2 2 --item-seq cut1 --num_steps 128 --log_interval 5" in terminal
# RL planner
This is a prototype of RL planner.

## Requirements

This project has been developed and tested with Python 3.8

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
pip install -r requirements.txt
```
## Training
```bash
python3 main.py --container_size 4 4 4 --item-size-range 2 2 2 2 2 2 --item-seq cut1 --num_steps 128 --log_interval 5
```
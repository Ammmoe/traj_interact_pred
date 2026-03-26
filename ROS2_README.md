# Setup to run in ROS2: Using official ROS 2 apt install + venv with `--system-site-packages`

## Install Ubuntu Python packages for ROS2 app

```bash
sudo apt update
sudo apt install -y \
  python3-venv \
  python3-pip \
  python3-pandas \
  python3-sklearn \
  python3-tqdm \
  python3-matplotlib
```

## Create the venv with the system Python.

```bash
python3 -m venv ~/venvs/fasttask --system-site-packages
```

## Activate the venv.

```bash
source ~/venvs/fasttask/bin/activate
```

## Install only the extra Python packages needed inside the venv.

```bash
python -m pip install -U pip
python -m pip install torch
```

## Add the followinh to ~/.bashrc

```bash
source ~/venvs/fasttask/bin/activate
export ROS2_INSTALL_PATH=/opt/ros/humble
source ${ROS2_INSTALL_PATH}/setup.bash
source ~/hifisim_ws/install/setup.bash
```

## Close the terminal and open a new terminal to re-run .bashrc

To check what interpreter is actually used

```bash
which python3
python3 -c "import sys; print(sys.executable)"
python3 -c "import rclpy; print(rclpy.__file__)"
python3 -c "import torch; print(torch.__version__)"
echo $VIRTUAL_ENV
```

## Important notes:

* Do not use `sudo pip install` into the system Python.




# Jetson Object Property

This repository is designed to analyze object properties using a Realsense camera and Docker containers on the NVIDIA Jetson Orin NX.

## Requirements

- **NVIDIA Jetson Orin NX** device
- Docker (for container-based execution)
- NVIDIA GPU support for Docker containers

## Setup Instructions

### 1. Run the Realsense Camera Node

Source the ROS 2 environment and launch the Realsense camera:

```bash
source /opt/ros/<ROS_DISTRO>/setup.bash
ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true pointcloud.enable:=true
```

### 2. Pull the Docker Container

Pull the NVIDIA container required for object property analysis:

```bash
docker pull nvcr.io/nvidia/jps/jps-gdino:ds7.1-public-12-11-1
```

### 3. Run the Container

Run the Docker container with the necessary volume mappings and GPU support:

```bash
docker run -itd -v /home/sastra/Desktop/ObjectProperty/output:/ds_microservices/output --runtime nvidia --network host nvcr.io/nvidia/jps/jps-gdino:ds7.1-public-12-11-1
```

### 4. Start Ollama Server for Image Segmentation

Run the Ollama server to handle the segmentation task:

```bash
docker run -d --runtime nvidia --name ollama-server -v ~/ollama:/ollama -e OLLAMA_MODELS=/ollama -p 11434:11434 dustynv/ollama:0.4.0-r36.4.0 ollama serve
```

## Notes

- This setup is designed to work on **NVIDIA Jetson Orin NX**.

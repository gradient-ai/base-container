In an attempt to make the leanest functional container we start with a bare Ubuntu 20.10 and load only the minimal requirements for ML processing with latest tensorflow running on CUDA 11.0.

Being bare bone, launching this container requires passing the `NVIDIA_VISIBLE_DEVICES=all` and the `NVIDIA_DRIVER_CAPABILITIES=compute,utility` env variables to the container on start, in addition to nvidia runtime.

For more info see: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#driver-capabilities

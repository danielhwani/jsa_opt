docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t joint_sa .

docker run \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -p 7860:7860 \
  -v "${PWD}:/workspace" \
  --shm-size=8G \
  -it joint_sa
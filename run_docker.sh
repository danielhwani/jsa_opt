docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t joint_sa .
docker run \
	--rm \
	--gpus all \
	-v ${PWD}:/workspace \
	--shm-size=32G \
	-it joint_sa;
	

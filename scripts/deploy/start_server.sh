#!/bin/bash

# Log in to ECR Mumbai
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 725813536964.dkr.ecr.ap-south-1.amazonaws.com

# Pull the latest image
docker pull 725813536964.dkr.ecr.ap-south-1.amazonaws.com/yt-chrome-plugin:latest

# Run the new container in the background
docker run -d -p 5000:5000 --name yt-chrome-plugin-container 725813536964.dkr.ecr.ap-south-1.amazonaws.com/yt-chrome-plugin:latest
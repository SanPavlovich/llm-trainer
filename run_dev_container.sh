#!/bin/bash
docker run -d -v C:/vscode_projects/complete/LLM:/root/LLM -it --rm --name pytorch --gpus all --privileged ubuntu-torch:latest bash
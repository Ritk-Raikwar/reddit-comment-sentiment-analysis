
#!/bin/bash
# Ignore errors if the container doesn't exist yet
docker stop yt-chrome-plugin-container || true
docker rm yt-chrome-plugin-container || true
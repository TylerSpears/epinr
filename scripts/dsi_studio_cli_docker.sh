#!/bin/bash

set -eou pipefail

TMP_DATA_DIR="${TMP_DATA_DIR:-/srv/tmp}"
DATA_DIR="${DATA_DIR:-/srv/data}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/.local/share/dsi-studio}"
DSI_CONFIG_DIR="${DSI_CONFIG_DIR:-$HOME/.config/dsi-studio}"
DSI_CACHE_DIR="${DSI_CACHE_DIR:-$HOME/.cache/dsi-studio}"
MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-$HOME/.cache/mesa_shader_cache}"
DSI_WORKDIR="${DSI_WORKDIR:-$(pwd)}"
mkdir --parents "$DSI_CONFIG_DIR" "$DSI_CACHE_DIR" "$LOCAL_DIR" "$MESA_SHADER_CACHE_DIR"

RUNTIME="${RUNTIME:-runc}"
IMAGE="${IMAGE:-dsistudio/dsistudio}"
TAG="${TAG:-hou-2025-04-17}"

docker run --rm \
    --ipc=host \
    --env HOME="/home/guest" \
    --env DISPLAY=$DISPLAY \
    --user "${UID}:${GROUPS}" \
    --runtime=$RUNTIME \
    --use-api-socket \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume "${DSI_WORKDIR}":"${DSI_WORKDIR}" \
    --workdir="$DSI_WORKDIR" \
    --volume "$DSI_CONFIG_DIR":/home/guest/.config \
    --volume "$DSI_CACHE_DIR":/home/guest/.cache \
    --volume "$LOCAL_DIR":/home/guest/.local/share/dsi-studio \
    --volume "$TMP_DATA_DIR":/srv/tmp \
    --volume "$DATA_DIR":/srv/data \
    "${IMAGE}:$TAG" \
    $@

# Reproduces the exact environment from pixi.lock and runs the reward comparison.
#
# The image ships CPU PyTorch only. This project's bottleneck is the environment step
# (highway-env physics plus two 256-particle filters per step), not the 128-unit MLPs, so
# a GPU buys very little here and pulling CUDA would triple the image size. If you do want
# the GPU, run with `--gpus all` on a CUDA base image and add a cuda pytorch entry to
# pixi.toml; nothing in the code assumes a device.
FROM ghcr.io/prefix-dev/pixi:0.41.4 AS build

WORKDIR /app

# Copy only what the solve needs first, so editing source does not invalidate the
# (very slow) dependency install layer.
COPY pixi.toml pixi.lock pyproject.toml ./
COPY src/negotiation/__init__.py src/negotiation/__init__.py
RUN pixi install --locked

COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY tests/ tests/

# Bake the activation into a plain shell script so the runtime stage does not need pixi
# on the PATH to enter the environment.
RUN pixi shell-hook -e default > /shell-hook.sh && \
    echo 'exec "$@"' >> /shell-hook.sh


FROM ubuntu:24.04 AS runtime

WORKDIR /app
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /shell-hook.sh /shell-hook.sh
COPY --from=build /app/src /app/src
COPY --from=build /app/scripts /app/scripts
COPY --from=build /app/configs /app/configs
COPY --from=build /app/tests /app/tests
COPY --from=build /app/pyproject.toml /app/pixi.toml /app/pixi.lock /app/

# highway-env imports pygame, which wants a display even when only doing physics.
ENV SDL_VIDEODRIVER=dummy
ENV MPLBACKEND=Agg

ENTRYPOINT ["/bin/bash", "/shell-hook.sh"]

# Default: the short paired comparison that produces results/*.json. Override with e.g.
#   docker run --rm -v "$PWD/results:/app/results" IMAGE pytest tests -q
#   docker run --rm IMAGE python scripts/train.py --config configs/default.yaml
CMD ["python", "scripts/run_comparison.py", "--total-steps", "20000", "--seeds", "0", \
     "--device", "cpu", "--tag", "docker_short"]

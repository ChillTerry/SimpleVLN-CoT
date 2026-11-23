"""Launch the SimpleVLN habitat env_server.

Takes the habitat config. The training parquet is loaded as one episode set.

Example:

  python -m recipe.vln_simplevln.env_server.launch \\
      --config recipe/vln_simplevln/config/vln_datasets.yaml \\
      --port 8002 --pool-size 8 --gpu-ids 0,1,2,3,4,5,6,7

Each worker runs one Habitat process. When GPU IDs are supplied, they cycle
through the worker pool. Use ``--cpu`` to run the recipe with Mesa llvmpipe.
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import uvicorn  # noqa: E402

from recipe.vln_simplevln.env_server.server import app, set_config  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--config", required=True,
                   help="Habitat config yaml.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--pool-size", type=int, default=1,
                   help="Number of Habitat worker processes. Each serves one active episode.")
    p.add_argument("--session-ttl-sec", type=float, default=1800.0)
    renderer = p.add_mutually_exclusive_group()
    renderer.add_argument("--cpu", action="store_true",
                          help="Use Mesa llvmpipe for CPU rendering with the VLN Habitat config.")
    renderer.add_argument("--gpu-ids", type=str,
                          help="Comma-separated GPU IDs assigned round-robin to workers (e.g. 0,1).")
    args = p.parse_args()

    gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else None
    if args.cpu:
        os.environ.update({
            "VLN_SIM_GPU_DEVICE_ID": "-1",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "GALLIUM_DRIVER": "llvmpipe",
            "CUDA_VISIBLE_DEVICES": "",
        })

    set_config(args.config, pool_size=args.pool_size,
               session_ttl_sec=args.session_ttl_sec, gpu_ids=gpu_ids)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

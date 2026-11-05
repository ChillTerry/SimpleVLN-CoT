"""Run ms-swift SFT with the inherited PPU runtime."""

import torch.distributed as dist
from ppu_compat import apply_zero2_accumulation_fix
from swift.llm.train.sft import SwiftSft


if __name__ == "__main__":
    apply_zero2_accumulation_fix()
    SwiftSft().main()
    if dist.is_initialized():
        dist.destroy_process_group()

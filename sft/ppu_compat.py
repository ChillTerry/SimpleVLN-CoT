"""Narrow, source-checked compatibility fixes for the inherited PPU runtime."""

import hashlib
import inspect
import textwrap

BAD_ACCUMULATION_CONDITION = "if not i in self.averaged_gradients or self.averaged_gradients[i] is None:"
FIXED_ACCUMULATION_CONDITION = "if i not in self.all_grad_tensors or self.all_grad_tensors[i] is None:"
VERIFIED_SOURCE_SHA256 = "8aab6ad622662a54879f409636e8fef248b6ab42d51cf69ab2131f5caa99d542"


def corrected_accumulation_source(source):
    """The persistent accumulator, not the boundary-only output, determines initialization."""
    if source.count(BAD_ACCUMULATION_CONDITION) != 1:
        return None
    return textwrap.dedent(source.replace(BAD_ACCUMULATION_CONDITION, FIXED_ACCUMULATION_CONDITION))


def apply_zero2_accumulation_fix():
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    previous = getattr(DeepSpeedZeroOptimizer, "_simplevln_accumulation_fix", None)
    if previous:
        return previous
    original = DeepSpeedZeroOptimizer.independent_gradient_partition_epilogue
    source = inspect.getsource(original)
    corrected = corrected_accumulation_source(source)
    if corrected is None:
        return {"applied": False, "reason": "Affected condition is absent"}
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if source_hash != VERIFIED_SOURCE_SHA256:
        raise RuntimeError("Unverified DeepSpeed epilogue source; rerun the gradient regression before adapting this fix")
    namespace = {}
    # Compile the installed, trusted method with only the checked condition
    # replaced. Vendor files and other optimizer behavior remain untouched.
    exec(compile(corrected, "<simplevln-zero2-accumulation>", "exec"), original.__globals__, namespace)  # noqa: S102 - checked installed source
    DeepSpeedZeroOptimizer.independent_gradient_partition_epilogue = namespace[original.__name__]
    metadata = {"applied": True, "revision": 1,
                "original_sha256": source_hash,
                "corrected_sha256": hashlib.sha256(corrected.encode()).hexdigest()}
    DeepSpeedZeroOptimizer._simplevln_accumulation_fix = metadata
    return metadata

#!/usr/bin/env python3
"""vLLM main: let `--kv-cache-dtype nvfp4_ds_mla` run on sm_120 through FlashInfer's DSv4.1 dual-cache
sparse MLA (flashinfer-ai/flashinfer#5197, in the FlashInfer nightly wheels since 0.7.0.dev20260919).

This is our upstream PR (branch `dsv41-sm120-nvfp4-flashinfer`, internal/upstream-dsv41-sm120-nvfp4-PR.md)
as an idempotent, anchor-based patcher for the vllm-openai:nightly based image, applied until the PR is
merged. Three files:

  vllm/utils/flashinfer.py                     has_flashinfer_sparse_mla_sm120_dsv41_records(): the installed
                                               FlashInfer accepts kv_cache_format="fp8_dsv41_fp4_ca"
  vllm/models/deepseek_v41/attention.py        _use_v41_mxfp8_kv_record() -> _use_v41_kv_records(kv_cache_dtype):
                                               SM100 as before; sm_120 takes the V4.1 records (528 B SWA + 288 B
                                               NVFP4 compressed) only for nvfp4_ds_mla and only when FlashInfer
                                               reads them; the indexer page alignment follows the same decision
  vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py
                                               the SM120 backend lists/gates nvfp4_ds_mla; the attention layer
                                               passes kv_cache_format="fp8_dsv41_fp4_ca" for dual-cache calls and
                                               "fp8_dsv41" for SWA-only calls (compress_ratio 0 layers, DSpark
                                               drafter); fp8_ds_mla passes no keyword (older FlashInfer keeps working)

With this patch the launcher's KV_RECORD=nvfp4 becomes `--kv-cache-dtype nvfp4_ds_mla` (KV_MODE=upstream) and
neither the fp8_ds_mla scratch/pool of patch_vllm_packed_kv_sm120.py nor the FlashInfer TU/hook of
patch_flashinfer.py are needed: FlashInfer >= 0.7.0.dev20260919 dispatches page sizes / heads / top-k at runtime.

Usage: python3 patch_vllm_nvfp4_sm120_upstream.py [--vllm-dir DIR] [--check]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")

FLASHINFER_UTILS: list[tuple[str, str]] = [
    (
        '    dispatch = getattr(mod, "_DECODE_DSV4_DISPATCH", None) if mod else None\n'
        "    return dispatch is not None and (int(num_q_heads), int(top_k)) in dispatch\n"
        "\n"
        "\n"
        "@functools.cache\n"
        "def has_flashinfer_cutedsl() -> bool:\n",
        '    dispatch = getattr(mod, "_DECODE_DSV4_DISPATCH", None) if mod else None\n'
        "    return dispatch is not None and (int(num_q_heads), int(top_k)) in dispatch\n"
        "\n"
        "\n"
        "@functools.cache\n"
        "def has_flashinfer_sparse_mla_sm120_dsv41_records() -> bool:\n"
        '    """Return whether FlashInfer\'s SM120 sparse MLA reads DeepSeek-V4.1\'s own\n'
        "    KV records.\n"
        "\n"
        "    flashinfer-ai/flashinfer#5197 added the DSv4.1 dual cache to the SM120 /\n"
        "    SM121 sparse MLA: the 528-byte all-fp8 V4.1 record (UE8M0 scale per 32\n"
        "    dims) as the sliding-window cache and the 288-byte NVFP4 record as the\n"
        '    compressed cache, selected with ``kv_cache_format="fp8_dsv41"`` /\n'
        '    ``"fp8_dsv41_fp4_ca"`` on ``trtllm_batch_decode_sparse_mla_dsv4``. Earlier\n'
        "    releases only read the V4 record (584 bytes), so the keyword and its\n"
        "    accepted values are the capability check.\n"
        '    """\n'
        "    if not has_flashinfer_sparse_mla_sm120():\n"
        "        return False\n"
        "    import inspect\n"
        "\n"
        "    from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4\n"
        "\n"
        "    try:\n"
        "        parameter = inspect.signature(trtllm_batch_decode_sparse_mla_dsv4).parameters[\n"
        '            "kv_cache_format"\n'
        "        ]\n"
        "    except (KeyError, TypeError, ValueError):\n"
        "        return False\n"
        '    return "fp8_dsv41_fp4_ca" in str(parameter.annotation)\n'
        "\n"
        "\n"
        "@functools.cache\n"
        "def has_flashinfer_cutedsl() -> bool:\n",
    ),
]

ATTENTION: list[tuple[str, str]] = [
    (
        "def _use_v41_mxfp8_kv_record() -> bool:\n"
        "    return current_platform.is_device_capability_family(100)\n",
        "# [vllm-moet] sm_120 takes the V4.1 records for nvfp4_ds_mla when FlashInfer's\n"
        "# DSv4.1 dual-cache sparse MLA (flashinfer#5197) reads them; fp8_ds_mla keeps V4.\n"
        "def _use_v41_kv_records(kv_cache_dtype: CacheDType | None) -> bool:\n"
        "    if current_platform.is_device_capability_family(100):\n"
        "        return True\n"
        '    if kv_cache_dtype != "nvfp4_ds_mla":\n'
        "        return False\n"
        "    from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_dsv41_records\n"
        "\n"
        "    return (\n"
        "        current_platform.is_device_capability_family(120)\n"
        "        and has_flashinfer_sparse_mla_sm120_dsv41_records()\n"
        "    )\n",
    ),
    (
        "        self.kv_mxfp8 = _use_v41_mxfp8_kv_record()\n",
        "        self.kv_mxfp8 = _use_v41_kv_records(self.kv_cache_dtype)\n",
    ),
    (
        '                "nvfp4_ds_mla needs the V4.1 KV records, which FlashMLA "\n'
        '                "decodes only on SM100."\n',
        '                "nvfp4_ds_mla needs the V4.1 KV records, which FlashMLA decodes "\n'
        '                "on SM100 and FlashInfer\'s DSv4.1 dual-cache sparse MLA on "\n'
        '                "SM120 / SM121 (flashinfer-ai/flashinfer#5197)."\n',
    ),
    (
        "        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype in (\n"
        '            "fp8_ds_mla",\n'
        '            "nvfp4_ds_mla",\n'
        "        )\n"
        "        page_alignment = (\n"
        "            576 if uses_fp8_ds_mla_layout and not _use_v41_mxfp8_kv_record() else 512\n"
        "        )\n",
        "        cache_dtype = vllm_config.cache_config.cache_dtype\n"
        '        uses_fp8_ds_mla_layout = cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")\n'
        "        page_alignment = (\n"
        "            576\n"
        "            if uses_fp8_ds_mla_layout and not _use_v41_kv_records(cache_dtype)\n"
        "            else 512\n"
        "        )\n",
    ),
]

FLASHINFER_SPARSE: list[tuple[str, str]] = [
    (
        '        "fp8_e4m3",\n'
        '        "fp8_ds_mla",\n'
        "    ]\n"
        "\n"
        "    @staticmethod\n"
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        return [128]\n",
        '        "fp8_e4m3",\n'
        '        "fp8_ds_mla",\n'
        "        # SM120 with flashinfer-ai/flashinfer#5197: V4.1 fp8 SWA record +\n"
        "        # NVFP4 compressed record, read by FlashInfer's DSv4.1 dual cache.\n"
        '        "nvfp4_ds_mla",\n'
        "    ]\n"
        "\n"
        "    @staticmethod\n"
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        return [128]\n",
    ),
    (
        "        if device_capability.major == 12:\n"
        '            if kv_cache_dtype not in ("fp8", "fp8_e4m3", "fp8_ds_mla"):\n'
        '                return "kv_cache_dtype not supported"\n'
        "            from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120\n"
        "\n"
        "            if not has_flashinfer_sparse_mla_sm120():\n"
        "                return (\n"
        '                    "FLASHINFER_MLA_SPARSE_DSV4 SM120 requires FlashInfer\'s "\n'
        '                    "sparse MLA decode API"\n'
        "                )\n"
        "            return None\n",
        "        if device_capability.major == 12:\n"
        '            if kv_cache_dtype not in ("fp8", "fp8_e4m3", "fp8_ds_mla", "nvfp4_ds_mla"):\n'
        '                return "kv_cache_dtype not supported"\n'
        "            from vllm.utils.flashinfer import (\n"
        "                has_flashinfer_sparse_mla_sm120,\n"
        "                has_flashinfer_sparse_mla_sm120_dsv41_records,\n"
        "            )\n"
        "\n"
        "            if not has_flashinfer_sparse_mla_sm120():\n"
        "                return (\n"
        '                    "FLASHINFER_MLA_SPARSE_DSV4 SM120 requires FlashInfer\'s "\n'
        '                    "sparse MLA decode API"\n'
        "                )\n"
        "            if (\n"
        '                kv_cache_dtype == "nvfp4_ds_mla"\n'
        "                and not has_flashinfer_sparse_mla_sm120_dsv41_records()\n"
        "            ):\n"
        "                return (\n"
        '                    "nvfp4_ds_mla on SM120 requires FlashInfer\'s DSv4.1 dual-cache "\n'
        '                    "sparse MLA (flashinfer-ai/flashinfer#5197: "\n'
        "                    'kv_cache_format=\"fp8_dsv41_fp4_ca\")'\n"
        "                )\n"
        "            return None\n",
    ),
    (
        '                "Install a FlashInfer build containing "\n'
        '                "flashinfer-ai/flashinfer#4380."\n'
        "            )\n"
        "        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe(\n",
        '                "Install a FlashInfer build containing "\n'
        '                "flashinfer-ai/flashinfer#4380."\n'
        "            )\n"
        "        # [vllm-moet] FlashInfer's packed-cache format: fp8_ds_mla (V4 record) is the\n"
        "        # kernel's default and passes no keyword; nvfp4_ds_mla stores the V4.1 records\n"
        '        # and selects "fp8_dsv41_fp4_ca" (dual cache) / "fp8_dsv41" (SWA cache alone).\n'
        "        self._flashinfer_kv_cache_format_kwargs: dict[str, str] = {}\n"
        "        self._flashinfer_swa_only_kv_cache_format_kwargs: dict[str, str] = {}\n"
        '        if self.kv_cache_dtype == "nvfp4_ds_mla":\n'
        '            assert self.kv_mxfp8, "nvfp4_ds_mla implies the V4.1 sliding-window record"\n'
        "            self._flashinfer_kv_cache_format_kwargs = {\n"
        '                "kv_cache_format": "fp8_dsv41_fp4_ca"\n'
        "            }\n"
        "            self._flashinfer_swa_only_kv_cache_format_kwargs = {\n"
        '                "kv_cache_format": "fp8_dsv41"\n'
        "            }\n"
        "        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe(\n",
    ),
    (
        "    def _reserve_empty_forward_workspace(self) -> None:\n"
        "        self._get_workspace(\n"
        '            torch.device("cuda", torch.accelerator.current_device_index())\n'
        "        )\n"
        "\n"
        "    def _forward_sparse_impl(\n",
        "    def _reserve_empty_forward_workspace(self) -> None:\n"
        "        self._get_workspace(\n"
        '            torch.device("cuda", torch.accelerator.current_device_index())\n'
        "        )\n"
        "\n"
        "    def _kv_cache_format_kwargs(self, has_compressed_cache: bool) -> dict[str, str]:\n"
        "        if has_compressed_cache:\n"
        "            return self._flashinfer_kv_cache_format_kwargs\n"
        "        return self._flashinfer_swa_only_kv_cache_format_kwargs\n"
        "\n"
        "    def _forward_sparse_impl(\n",
    ),
    (
        "            swa_topk_lens=swa_lens,\n"
        "            extra_sparse_indices=extra_sparse_indices,\n"
        "            extra_sparse_topk_lens=extra_sparse_lengths,\n"
        "        )\n",
        "            swa_topk_lens=swa_lens,\n"
        "            extra_sparse_indices=extra_sparse_indices,\n"
        "            extra_sparse_topk_lens=extra_sparse_lengths,\n"
        "            **self._kv_cache_format_kwargs(extra_cache is not None),\n"
        "        )\n",
    ),
    (
        "                swa_topk_lens=swa_lens_chunk,\n"
        "                extra_sparse_indices=extra_sparse_indices_chunk,\n"
        "                extra_sparse_topk_lens=extra_sparse_lengths_chunk,\n"
        "            )\n",
        "                swa_topk_lens=swa_lens_chunk,\n"
        "                extra_sparse_indices=extra_sparse_indices_chunk,\n"
        "                extra_sparse_topk_lens=extra_sparse_lengths_chunk,\n"
        "                **self._kv_cache_format_kwargs(extra_kv_paged is not None),\n"
        "            )\n",
    ),
]

FILES = {
    "utils/flashinfer.py": FLASHINFER_UTILS,
    "models/deepseek_v41/attention.py": ATTENTION,
    "models/deepseek_v41/nvidia/flashinfer_sparse.py": FLASHINFER_SPARSE,
}


def apply(path: Path, edits: list[tuple[str, str]], check: bool) -> bool:
    s = path.read_text()
    if all(new in s for _, new in edits):
        print(f"{path}: already patched")
        return True
    for old, new in edits:
        if new in s:
            continue
        n = s.count(old)
        if n != 1:
            print(f"{path}: anchor x{n} != 1:\n{old[:160]}", file=sys.stderr)
            return False
        s = s.replace(old, new)
    if check:
        print(f"{path}: patch applies cleanly (not written)")
        return True
    compile(s, str(path), "exec")
    path.write_text(s)
    print(f"{path}: patched ({len(edits)} sites)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-dir", type=Path, default=DEFAULT_VLLM)
    ap.add_argument("--check", action="store_true", help="verify anchors, write nothing")
    args = ap.parse_args()
    ok = True
    for rel, edits in FILES.items():
        path = args.vllm_dir / rel
        if not path.exists():
            print(f"{path}: missing (vLLM main layout expected)", file=sys.stderr)
            ok = False
            continue
        ok = apply(path, edits, args.check) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

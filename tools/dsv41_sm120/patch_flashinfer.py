#!/usr/bin/env python3
"""Patch FlashInfer's SM120 sparse-MLA JIT sources for the DeepSeek-V4.1 geometry.

What is missing in stock FlashInfer 0.6.18 (the pin of vllm/vllm-openai:deepseekv41-flash-0909)
for DeepSeek-V4.1 on sm_120, and what this patcher adds:

  gap                                              | stock                         | added
  -------------------------------------------------|-------------------------------|-----------------------------
  SWA cache page (vLLM DeepseekV4SWACache = 32)    | decode + prefill PBS=64 only  | PBS=32 instantiations
  compressed cache page on ratio-1 layers (128)    | dual prefill PBSX in {64, 2}  | PBSX=128 (and 64) at PBS=32
  vision-padded prefill SWA rows (128+1024 = 1152) | TOPK in {128..2048}, dual 128 | TOPK=1152 single + dual
  DSpark non-causal decode rows (192) via orchestr.| dual TOPK=128 only            | dual TOPK=192
  single-cache dispatch ignored page_block_size    | PBS=64 kernel ran on 32-pages | DSV4 + PBS!=64 -> DSV4.1 table

Files touched (installed package, JIT sources):
  data/csrc/sparse_mla_sm120_dsv41.cu       NEW  - the instantiations + dispatcher (this dir)
  data/csrc/sparse_mla_sm120_prefill.cu     hook - route DSV4 with SWA page != 64 to the DSV4.1 table
  data/csrc/sparse_mla_sm120_decode_dsv4.cu      - PBS=32 decode instantiations (TOPK 128/192/256)
  jit/mla.py                                     - add the new TU to the sparse_mla_sm120 module
  mla/_sparse_mla_sm120.py                       - accept page 32 on the decode fast path

Idempotent; hard anchors (assert). Run against an installed flashinfer package:
  python3 patch_flashinfer.py [--flashinfer-dir /usr/local/lib/python3.12/dist-packages/flashinfer]
The AOT artifact flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so must be
removed (or made unloadable) afterwards, otherwise it is loaded INSTEAD of the patched sources.
"""
import argparse
import ast
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
MARK = "DSV4.1"


def patch(path: pathlib.Path, old: str, new: str, count: int = 1):
    s = path.read_text()
    if new in s:
        return
    assert old in s, f"ANCHOR NOT FOUND in {path}:\n{old[:200]}"
    assert s.count(old) == count, f"anchor x{s.count(old)} != {count} in {path}"
    path.write_text(s.replace(old, new))
    print(f"patched: {path.name}: {old.strip()[:60]!r}...")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--flashinfer-dir",
        default="/usr/local/lib/python3.12/dist-packages/flashinfer",
        help="installed flashinfer package directory",
    )
    args = ap.parse_args()
    fi = pathlib.Path(args.flashinfer_dir)
    csrc = fi / "data/csrc"
    assert (csrc / "sparse_mla_sm120_prefill.cu").exists(), f"not a flashinfer package: {fi}"

    # ------------------------------------------------------ 0. new TU
    src = HERE / "sparse_mla_sm120_dsv41.cu"
    dst = csrc / "sparse_mla_sm120_dsv41.cu"
    if not dst.exists() or dst.read_text() != src.read_text():
        shutil.copyfile(src, dst)
        print("installed: data/csrc/sparse_mla_sm120_dsv41.cu")

    # --------------------------------------------- 1. jit/mla.py sources
    patch(
        fi / "jit/mla.py",
        '''            jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120_prefill.cu",
            jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120_jit_binding.cu",''',
        '''            jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120_prefill.cu",
            # DSV4.1 (vLLM-Moet): SWA page 32 / compressed page 128 / 1152-wide rows.
            jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120_dsv41.cu",
            jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120_jit_binding.cu",''',
    )

    # ------------------------------------------ 2. prefill orchestrator hook
    pre = csrc / "sparse_mla_sm120_prefill.cu"
    patch(
        pre,
        """namespace flashinfer::sparse_mla_sm120 {

namespace {

constexpr int kMaxCachedCudaDevices = 32;""",
        """namespace flashinfer::sparse_mla_sm120 {

// DSV4.1 (vLLM-Moet, sparse_mla_sm120_dsv41.cu): SWA page 32, compressed page
// 128/64, SWA rows 128/192/1152. Consulted for ModelType::DSV4 whenever the
// SWA page is not the stock 64.
bool sparse_mla_prefill_dispatch_dsv41(int num_heads, int topk, int page_block_size,
                                       int topk_extra, int extra_page_block_size, const bf16* Q,
                                       const uint8_t* KV_cache, const int32_t* indices,
                                       const uint8_t* extra_KV_cache,
                                       const int32_t* extra_indices, bf16* output,
                                       float* out_lse, float sm_scale, int num_tokens,
                                       size_t stride_kv_block, size_t stride_kv_block_extra,
                                       const float* attn_sink, const int* topk_length,
                                       const int* extra_topk_length, cudaStream_t stream);

namespace {

constexpr int kMaxCachedCudaDevices = 32;""",
    )
    patch(
        pre,
        """                                 const int* extra_topk_length, cudaStream_t stream) {
  if (extra_KV_cache != nullptr) {
    if (mt != ModelType::DSV4) return false;""",
        """                                 const int* extra_topk_length, cudaStream_t stream) {
  // DSV4.1: the stock DSV4 tables below are PAGE_BLOCK_SIZE=64 instantiations
  // and dispatch_dsv4_single never looked at page_block_size, so any other SWA
  // page must take the DSV4.1 table (or fail loudly) instead of running a
  // 64-page kernel over 32-token pages.
  if (mt == ModelType::DSV4 && page_block_size != 64) {
    return sparse_mla_prefill_dispatch_dsv41(
        num_heads, topk, page_block_size, topk_extra, extra_page_block_size, Q, KV_cache, indices,
        extra_KV_cache, extra_indices, output, out_lse, sm_scale, num_tokens, stride_kv_block,
        stride_kv_block_extra, attn_sink, topk_length, extra_topk_length, stream);
  }
  if (extra_KV_cache != nullptr) {
    if (mt != ModelType::DSV4) return false;""",
    )

    # ------------------------------------------------- 3. decode dsv4 PBS=32
    dec = csrc / "sparse_mla_sm120_decode_dsv4.cu"
    patch(
        dec,
        "  if (mt != ModelType::DSV4 || page_block_size != 64) return false;",
        "  if (mt != ModelType::DSV4 || (page_block_size != 64 && page_block_size != 32)) return false;",
    )
    patch(
        dec,
        """#define DSV4_DISPATCH(H, K)                                                                 \\
  if (num_heads == (H) && topk == (K)) {                                                    \\
    return launch_decode_dsv4_impl<ModelType::DSV4, (H), (K), 64>(                          \\
        Q, KV_cache, indices, mid_out, mid_lse, topk_length, output, out_lse, attn_sink,    \\
        extra_KV_cache, extra_indices, extra_topk_length, extra_topk, pbs_extra,            \\
        stride_extra_kv_block, num_tokens, num_splits, chunks_per_block_override, sm_scale, \\
        stride_kv_block, stream);                                                           \\
  }
  DSV4_DISPATCH(8, 128)""",
        """#define DSV4_DISPATCH_PBS(H, K, PBS)                                                        \\
  if (num_heads == (H) && topk == (K) && page_block_size == (PBS)) {                        \\
    return launch_decode_dsv4_impl<ModelType::DSV4, (H), (K), (PBS)>(                       \\
        Q, KV_cache, indices, mid_out, mid_lse, topk_length, output, out_lse, attn_sink,    \\
        extra_KV_cache, extra_indices, extra_topk_length, extra_topk, pbs_extra,            \\
        stride_extra_kv_block, num_tokens, num_splits, chunks_per_block_override, sm_scale, \\
        stride_kv_block, stream);                                                           \\
  }
#define DSV4_DISPATCH(H, K) DSV4_DISPATCH_PBS(H, K, 64)
  // DSV4.1 (vLLM-Moet): SWA cache page 32; rows 128 (window), 192/256 (DSpark
  // non-causal widths), 1152 (vision-padded prefill rows of a <=64-token
  // batch, which vLLM routes through the decode entry). The extra
  // (compressed) page stays a runtime argument.
  DSV4_DISPATCH_PBS(8, 128, 32)
  DSV4_DISPATCH_PBS(8, 192, 32)
  DSV4_DISPATCH_PBS(8, 256, 32)
  DSV4_DISPATCH_PBS(8, 1152, 32)
  DSV4_DISPATCH_PBS(16, 128, 32)
  DSV4_DISPATCH_PBS(16, 192, 32)
  DSV4_DISPATCH_PBS(16, 256, 32)
  DSV4_DISPATCH_PBS(16, 1152, 32)
  DSV4_DISPATCH_PBS(32, 128, 32)
  DSV4_DISPATCH_PBS(32, 192, 32)
  DSV4_DISPATCH_PBS(32, 256, 32)
  DSV4_DISPATCH_PBS(32, 1152, 32)
  DSV4_DISPATCH_PBS(64, 128, 32)
  DSV4_DISPATCH_PBS(64, 192, 32)
  DSV4_DISPATCH_PBS(64, 256, 32)
  DSV4_DISPATCH_PBS(64, 1152, 32)
  DSV4_DISPATCH(8, 128)""",
    )
    patch(dec, "#undef DSV4_DISPATCH\n", "#undef DSV4_DISPATCH\n#undef DSV4_DISPATCH_PBS\n")

    # ------------------------------------------------ 4. python decode path
    py = fi / "mla/_sparse_mla_sm120.py"
    patch(
        py,
        "_DECODE_DSV4_PAGE_BLOCK_SIZE = 64\n",
        """_DECODE_DSV4_PAGE_BLOCK_SIZE = 64
# DSV4.1 (vLLM-Moet): the SWA cache is paged at 32 tokens; decode kernels for
# that page are instantiated for these (num_heads, topk) shapes.
_DECODE_DSV4_PAGE_BLOCK_SIZES = frozenset({64, 32})
_DECODE_DSV4_DISPATCH_PBS32 = frozenset(
    {(h, k) for h in (8, 16, 32, 64) for k in (128, 192, 256, 1152)}
)
""",
    )
    patch(
        py,
        """    return (
        num_tokens <= _DECODE_MAX_TOKENS
        and d_qk == 512
        and page_block_size == _DECODE_DSV4_PAGE_BLOCK_SIZE
        and (num_heads, topk) in _DECODE_DSV4_DISPATCH
    )""",
        """    if page_block_size == 32:
        return (
            num_tokens <= _DECODE_MAX_TOKENS
            and d_qk == 512
            and (num_heads, topk) in _DECODE_DSV4_DISPATCH_PBS32
        )
    return (
        num_tokens <= _DECODE_MAX_TOKENS
        and d_qk == 512
        and page_block_size == _DECODE_DSV4_PAGE_BLOCK_SIZE
        and (num_heads, topk) in _DECODE_DSV4_DISPATCH
    )""",
    )
    patch(
        py,
        """        if (
            model_type == _MODEL_TYPE_DSV4
            and kv_pbs == _DECODE_DSV4_PAGE_BLOCK_SIZE
            and _decode_dsv4_dispatchable(""",
        """        if (
            model_type == _MODEL_TYPE_DSV4
            and kv_pbs in _DECODE_DSV4_PAGE_BLOCK_SIZES
            and _decode_dsv4_dispatchable(""",
    )
    ast.parse(py.read_text())
    ast.parse((fi / "jit/mla.py").read_text())
    print("SYNTAX-OK python")
    print("PATCH-FLASHINFER-DSV41-DONE")


if __name__ == "__main__":
    main()

from typing import Any, Dict

import torch


def quantize_rows_fp8_triton(
    input: torch.Tensor,
    output: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    import triton

    from .triton.fused_moe import quantize_rows_fp8_kernel

    input_2d = input.view(-1, input.shape[-1])
    output_2d = output.view_as(input_2d)
    if output_2d.dtype != torch.float8_e4m3fn:
        raise ValueError("FP8 activation output must use torch.float8_e4m3fn")
    if scale.shape != (input_2d.shape[0],):
        raise ValueError("FP8 activation scale must contain one value per row")
    block_size = triton.next_power_of_2(input_2d.shape[1])
    quantize_rows_fp8_kernel[(input_2d.shape[0],)](
        input_2d,
        output_2d,
        scale,
        input_2d.shape[1],
        input_2d.stride(0),
        input_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )


def direct_moe_gemv_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    B_scale: torch.Tensor | None = None,
    *,
    input_is_routed: bool,
    mul_routed_weight: bool,
    config: Dict[str, Any],
) -> None:
    import triton

    from .triton.fused_moe import direct_moe_gemv_kernel

    num_routes = topk_ids.numel()
    top_k = topk_ids.shape[1]
    output_size = B.shape[1]
    reduction_size = B.shape[2]
    output_2d = C.view(num_routes, output_size)
    input_2d = A.view(-1, reduction_size)
    grid = (num_routes, triton.cdiv(output_size, config["BLOCK_SIZE_N"]))
    direct_moe_gemv_kernel[grid](
        input_2d,
        B,
        output_2d,
        topk_weights,
        topk_ids,
        B if B_scale is None else B_scale,
        num_routes,
        B.shape[0],
        output_size,
        reduction_size,
        input_2d.stride(0),
        input_2d.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        output_2d.stride(0),
        output_2d.stride(1),
        TOP_K=top_k,
        INPUT_IS_ROUTED=input_is_routed,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        HAS_SCALE=B_scale is not None,
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
    )


def fused_moe_kernel_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: torch.dtype,
    B_scale: torch.Tensor | None = None,
    A_scale: torch.Tensor | None = None,
) -> None:
    import triton
    import triton.language as tl

    from .triton.fused_moe import fused_moe_kernel

    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    padded_size = 0
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False
    dtype = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }[compute_type]
    native_fp8 = A.dtype == torch.float8_e4m3fn or B.dtype == torch.float8_e4m3fn
    if native_fp8 and not (
        A.dtype == torch.float8_e4m3fn
        and B.dtype == torch.float8_e4m3fn
        and A_scale is not None
        and B_scale is not None
    ):
        raise ValueError("Native FP8 MoE requires FP8 A/B tensors and both scale tensors")
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        B if B_scale is None else B_scale,
        A if A_scale is None else A_scale,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2] - padded_size,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore
        top_k=top_k,  # type: ignore
        compute_type=dtype,  # type: ignore
        even_Ks=even_Ks,  # type: ignore
        HAS_SCALE=B_scale is not None,
        NATIVE_FP8=native_fp8,
        **config,
    )


def moe_sum_reduce_triton(input: torch.Tensor, output: torch.Tensor) -> None:
    import triton

    from .triton.fused_moe import moe_sum_reduce_kernel

    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 8

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,  # type: ignore
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,  # type: ignore
    )

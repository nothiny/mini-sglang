import triton
import triton.language as tl


@triton.jit
def quantize_rows_fp8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    N,
    stride_input_m,
    stride_input_n,
    stride_output_m,
    stride_output_n,
    BLOCK_SIZE: tl.constexpr,
):
    """Quantize each activation row to E4M3 with one FP32 scale."""

    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        input_ptr + row * stride_input_m + offsets * stride_input_n,
        mask=offsets < N,
        other=0.0,
    ).to(tl.float32)
    abs_max = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(abs_max / 448.0, 1e-8)
    quantized = tl.maximum(tl.minimum(values / scale, 448.0), -448.0)
    tl.store(scale_ptr + row, scale)
    tl.store(
        output_ptr + row * stride_output_m + offsets * stride_output_n,
        quantized,
        mask=offsets < N,
    )


@triton.jit
def direct_moe_gemv_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    b_scale_ptr,
    num_routes,
    num_experts,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    TOP_K: tl.constexpr,
    INPUT_IS_ROUTED: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Decode-oriented expert GEMV without expert sorting or block padding."""

    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    expert_id = tl.load(topk_ids_ptr + route_id)
    route_valid = (route_id < num_routes) & (expert_id >= 0) & (expert_id < num_experts)
    safe_expert_id = tl.minimum(tl.maximum(expert_id, 0), num_experts - 1)
    input_row = route_id if INPUT_IS_ROUTED else route_id // TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for k_block_id in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        current_k = k_block_id * BLOCK_SIZE_K + offs_k
        a = tl.load(
            a_ptr + input_row * stride_am + current_k * stride_ak,
            mask=route_valid & (current_k < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + safe_expert_id * stride_be
            + offs_n[:, None] * stride_bn
            + current_k[None, :] * stride_bk,
            mask=route_valid & (offs_n[:, None] < N) & (current_k[None, :] < K),
            other=0.0,
        )
        if HAS_SCALE:
            scale = tl.load(
                b_scale_ptr + safe_expert_id * N + offs_n,
                mask=route_valid & (offs_n < N),
                other=0.0,
            )
            b = b.to(tl.float32) * scale[:, None]
        accumulator += tl.sum(b.to(tl.float32) * a[None, :].to(tl.float32), axis=1)

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(topk_weights_ptr + route_id, mask=route_valid, other=0.0)
        accumulator *= routed_weight

    tl.store(
        c_ptr + route_id * stride_cm + offs_n * stride_cn,
        accumulator,
        mask=offs_n < N,
    )


@triton.jit
def moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0)
            accumulator += tmp
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    b_scale_ptr,
    a_scale_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        if HAS_SCALE and not NATIVE_FP8:
            scale = tl.load(
                b_scale_ptr + off_experts * N + offs_bn,
                mask=offs_bn < N,
                other=0.0,
            )
            b = (b.to(tl.float32) * scale[None, :]).to(compute_type)

        # We accumulate along the K dimension.

        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if NATIVE_FP8:
        input_scale = tl.load(
            a_scale_ptr + offs_token // top_k,
            mask=token_mask,
            other=0.0,
        )
        weight_scale = tl.load(
            b_scale_ptr + off_experts * N + offs_bn,
            mask=offs_bn < N,
            other=0.0,
        )
        accumulator *= input_scale[:, None] * weight_scale[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_elements", "num_layers", "pool_pages", "page_elements"])
def hicache_gather_pages_kernel(
    source_ptr,
    page_ids_ptr,
    packed_ptr,
    num_elements,
    num_layers,
    pool_pages,
    page_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    inner = offsets % page_elements
    logical_page = offsets // (page_elements * num_layers * 2)
    layer_kv = (offsets // page_elements) % (num_layers * 2)
    physical_page = tl.load(page_ids_ptr + logical_page, mask=mask, other=0).to(tl.int64)
    source_offsets = (layer_kv * pool_pages + physical_page) * page_elements + inner
    values = tl.load(source_ptr + source_offsets, mask=mask)
    tl.store(packed_ptr + offsets, values, mask=mask)


@triton.jit(do_not_specialize=["num_elements", "num_layers", "pool_pages", "page_elements"])
def hicache_scatter_pages_kernel(
    packed_ptr,
    page_ids_ptr,
    destination_ptr,
    num_elements,
    num_layers,
    pool_pages,
    page_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    inner = offsets % page_elements
    logical_page = offsets // (page_elements * num_layers * 2)
    layer_kv = (offsets // page_elements) % (num_layers * 2)
    physical_page = tl.load(page_ids_ptr + logical_page, mask=mask, other=0).to(tl.int64)
    destination_offsets = (layer_kv * pool_pages + physical_page) * page_elements + inner
    values = tl.load(packed_ptr + offsets, mask=mask)
    tl.store(destination_ptr + destination_offsets, values, mask=mask)

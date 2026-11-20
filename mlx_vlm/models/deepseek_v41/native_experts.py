"""Rank-local matvecs for the original packed MXFP4 expert weights."""

from functools import lru_cache

import mlx.core as mx

from ..masked_quantized import _HEADER
from ..switch_layers import QuantizedSwitchLinear


def _supported(layer):
    return (
        type(layer) is QuantizedSwitchLinear
        and layer.mode == "mxfp4"
        and layer.bits == 4
        and layer.group_size == 32
        and layer.weight.dtype == mx.uint32
        and layer.scales.dtype == mx.uint8
        and layer.input_dims % 256 == 0
        and layer.output_dims % 8 == 0
        and layer.get("biases") is None
        and "bias" not in layer
    )


@lru_cache(maxsize=2)
def _projection_kernel(paired):
    # Keep MLX's eight- or sixteen-element dot groups, FP32 accumulation and
    # per-projection BF16 rounding. Gate and up share their activation loads.
    per_row = (
        """
        second[i] += qdot<float,P,4>(gate_weight+wi, xv,
                                     dequantize_scale<float,32>(gate_scale[si]));
    """
        if paired
        else ""
    )
    gate_setup = (
        """
        const device uint8_t* gate_weight = (const device uint8_t*)wg + base/2 + lane*(P/2);
        const device uint8_t* gate_scale = sg + base/32 + lane*P/32;
        thread float second[4] = {0};
    """
        if paired
        else ""
    )
    gate_write = (
        """
        const float gate_value = simd_sum(second[i]);
        if (lane == 0) gate[route*N+row+i] = T(gate_value);
    """
        if paired
        else ""
    )
    clear_gate = "gate[route*N+row+i] = T(0);" if paired else ""
    return mx.fast.metal_kernel(
        name="deepseek_v41_mxfp4_gate_up" if paired else "deepseek_v41_mxfp4_down",
        input_names=["x", "indices", "wu", "su"] + (["wg", "sg"] if paired else []),
        output_names=["up", "gate"] if paired else ["up"],
        header=_HEADER,
        compile_options={"math_mode": "safe"},
        source=f"""
        const uint route = threadgroup_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        const uint row = threadgroup_position_in_grid.y*8 + simdgroup_index_in_threadgroup*4;
        const int expert = indices[route];
        if (expert < 0) {{
            if (lane == 0) for (int i=0; i<4; ++i) {{
                up[route*N+row+i] = T(0);
                {clear_gate}
            }}
            return;
        }}
        constexpr int P = K%512 == 0 ? 16 : 8;
        const size_t base = (size_t(expert)*N+row)*K;
        const device uint8_t* weight = (const device uint8_t*)wu + base/2 + lane*(P/2);
        const device uint8_t* scale = su + base/32 + lane*P/32;
        const device T* input = x + (route/FAN)*K + lane*P;
        thread float accum[4] = {{0}};
        {gate_setup}
        for (int k=0; k<x_shape[x_ndim-1]; k+=32*P) {{
            thread float xv[P];
            load_vector<T,float,P>(input+k, xv);
            for (int i=0; i<4; ++i) {{
                const size_t wi=(k+i*K)/2, si=(k+i*K)/32;
                accum[i] += qdot<float,P,4>(weight+wi, xv,
                                            dequantize_scale<float,32>(scale[si]));
                {per_row}
            }}
        }}
        for (int i=0; i<4; ++i) {{
            const float value = simd_sum(accum[i]);
            if (lane == 0) up[route*N+row+i] = T(value);
            {gate_write}
        }}
        """,
    )


def _project(x, indices, fan, layer, gate=None):
    k, n = layer.input_dims, layer.output_dims
    inputs = [x, indices, layer.weight, layer.scales]
    if gate is not None:
        inputs.extend([gate.weight, gate.scales])
    outputs = 2 if gate is not None else 1
    return _projection_kernel(gate is not None)(
        inputs=inputs,
        template=[("T", x.dtype), ("K", k), ("N", n), ("FAN", fan)],
        grid=(32, n // 8 * 2, indices.size),
        threadgroup=(32, 2, 1),
        output_shapes=[(*indices.shape, n)] * outputs,
        output_dtypes=[x.dtype] * outputs,
    )


def local_experts(glu, x, indices, weights, start):
    """Return the local routed sum, or None for an unsupported layout."""
    if (
        x.dtype != mx.bfloat16
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or not all(_supported(p) for p in (glu.up_proj, glu.gate_proj, glu.down_proj))
    ):
        return None
    local = indices.astype(mx.int32) - start
    local = mx.where((local >= 0) & (local < glu.up_proj.num_experts), local, -1)
    up, gate = _project(x, local, indices.shape[-1], glu.up_proj, glu.gate_proj)
    activated = glu.activation(up, gate, weights[..., None])
    (down,) = _project(activated, local, 1, glu.down_proj)
    return down.astype(mx.float32).sum(-2)

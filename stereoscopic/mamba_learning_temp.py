def mamba_inner:
    # split xz
    x, z = xz.chunk(2, dim=1)
    # Compute short convolution
    if conv_state is not None:
        # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
        # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
        conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
    if causal_conv1d_fn is None:
        x = self.act(self.conv1d(x)[..., :seqlen])
    else:
        assert self.activation in ["silu", "swish"]
        x = causal_conv1d_fn(
            x=x,
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=self.activation,
        )

    # We're careful here about the layout, to avoid extra transposes.
    # We want dt to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
    dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    dt = self.dt_proj.weight @ dt.t()
    dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
    B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    assert self.activation in ["silu", "swish"]
    y = selective_scan_fn(
        x,
        dt,
        A,
        B,
        C,
        self.D.float(),
        z=z,
        delta_bias=self.dt_proj.bias.float(),
        delta_softplus=True,
        return_last_state=ssm_state is not None,
    )
    if ssm_state is not None:
        y, last_state = y
        ssm_state.copy_(last_state)
    y = rearrange(y, "b d l -> b l d")
    out = self.out_proj(y)

def mamba_inner_fastpath:
    assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
    assert checkpoint_lvl in [0, 1]
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    if torch.is_autocast_enabled():
        x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
        delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
        out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
        out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                         if out_proj_bias is not None else None)
    if xz.stride(-1) != 1:
        xz = xz.contiguous()
    conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")

    # split xz
    x, z = xz.chunk(2, dim=1)
    conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
    conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
        x, conv1d_weight, conv1d_bias, None, None, None, True
    )

    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
    ctx.is_variable_B = B is None
    ctx.is_variable_C = C is None
    ctx.B_proj_bias_is_None = B_proj_bias is None
    ctx.C_proj_bias_is_None = C_proj_bias is None
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
    else:
        if B.stride(-1) != 1:
            B = B.contiguous()
    if C is None:  # variable C
        C = x_dbl[:, -d_state:]  # (bl dstate)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
    else:
        if C.stride(-1) != 1:
            C = C.contiguous()
    if D is not None:
        D = D.contiguous()
    out, scan_intermediates, out_z = selective_scan_cuda.fwd(
        conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
    )
    ctx.delta_softplus = delta_softplus
    ctx.out_proj_bias_is_None = out_proj_bias is None
    ctx.checkpoint_lvl = checkpoint_lvl
    if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
        conv1d_out, delta = None, None
    ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                          delta_proj_weight, out_proj_weight, conv1d_out, delta,
                          A, B, C, D, delta_bias, scan_intermediates, out)
    return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
import torch
import torch.nn.functional as F
from typing import Literal, Tuple, Optional
import os, sys
import math

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(os.path.join(ROOT, "RVT"))

from models.layers.s5.jax_func import associative_scan
from models.layers.s5.s5_init import *

# Runtime functions


@torch.jit.script
def binary_operator(
    q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]
):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    # return A_j * A_i, A_j * b_i + b_j
    return A_j * A_i, torch.addcmul(b_j, A_j, b_i)

from einops import einsum

def apply_ssm(
    Lambda_bars: torch.Tensor,
    B_bars,
    C_tilde,
    D,
    input_sequence,
    prev_state,
    bidir: bool = False,
):
    B_bars = as_complex(B_bars)
    C_tilde = as_complex(C_tilde)
    Lambda_bars = as_complex(Lambda_bars)

    cinput_sequence = input_sequence.type(
        Lambda_bars.dtype
    )  # Cast to correct complex type

    if B_bars.ndim == 3:
        # Dynamic timesteps (significantly more expensive)
        Bu_elements = torch.vmap(lambda B_bar, u: B_bar @ u)(B_bars, cinput_sequence)
    else:
        # Static timesteps
        Bu_elements = torch.vmap(lambda u: B_bars @ u)(cinput_sequence)

    if Lambda_bars.ndim == 1:  # Repeat for associative_scan
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)

    Lambda_bars[0] = Lambda_bars[0] * prev_state

    _, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))

    if bidir:
        _, xs2 = associative_scan(
            binary_operator, (Lambda_bars, Bu_elements), reverse=True
        )
        xs = torch.cat((xs, xs2), axis=-1)

    Du = torch.vmap(lambda u: D * u)(input_sequence)
    # TODO: the last element of xs (non-bidir) is the hidden state, allow returning it
    return torch.vmap(lambda x: (C_tilde @ x).real)(xs) + Du, xs[-1]


def apply_ssm_liquid(
    Lambda_bars, B_bars, C_tilde, D, input_sequence, bidir: bool = False
):
    """Liquid time constant SSM \u00e1 la dynamical systems given in Eq. 8 of
    https://arxiv.org/abs/2209.12951"""
    cinput_sequence = input_sequence.type(
        Lambda_bars.dtype
    )  # Cast to correct complex type

    if B_bars.ndim == 3:
        # Dynamic timesteps (significantly more expensive)
        Bu_elements = torch.vmap(lambda B_bar, u: B_bar @ u)(B_bars, cinput_sequence)
    else:
        # Static timesteps
        Bu_elements = torch.vmap(lambda u: B_bars @ u)(cinput_sequence)

    if Lambda_bars.ndim == 1:  # Repeat for associative_scan
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)

    _, xs = associative_scan(binary_operator, (Lambda_bars + Bu_elements, Bu_elements))

    if bidir:
        _, xs2 = associative_scan(
            binary_operator, (Lambda_bars, Bu_elements), reverse=True
        )
        xs = torch.cat((xs, xs2), axis=-1)

    Du = torch.vmap(lambda u: D * u)(input_sequence)
    return torch.vmap(lambda x: (C_tilde @ x).real)(xs) + Du

def ssm(self, x):
    """Runs the SSM. See:
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]

    Args:
        x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

    Returns:
        output: shape (b, l, d_in)

    Official Implementation:
        mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        
    """
    (d_in, n) = self.A_log.shape

    # Compute ∆ A B C D, the state space parameters.
    #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    #                                  and is why Mamba is called **selective** state spaces)
    
    A = -torch.exp(self.A_log.float())  # shape (d_in, n)
    D = self.D.float()

    x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
    
    (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
    delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
    
    y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
    
    return y


def selective_scan(u, delta, A, B, C, D):
    """Does selective scan algorithm. See:
        - Section 2 State Space Models in the Mamba paper [1]
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]

    This is the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

    Args:
        u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
        delta: shape (b, l, d_in)
        A: shape (d_in, n)
        B: shape (b, l, n)
        C: shape (b, l, n)
        D: shape (d_in,)

    Returns:
        output: shape (b, l, d_in)

    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
        Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
        
    """
    import pdb;pdb.set_trace()
    (b, l, d_in) = u.shape
    n = A.shape[1]
    
    # Discretize continuous parameters (A, B)
    # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
    # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
    #   "A is the more important term and the performance doesn't change much with the simplification on B"
    deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
    deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
    
    # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
    # Note that the below is sequential, while the official implementation does a much faster parallel scan that
    # is additionally hardware-aware (like FlashAttention).
    x = torch.zeros((b, d_in, n), device=deltaA.device)
    ys = []    
    for i in range(l):
        x = deltaA[:, i] * x + deltaB_u[:, i]
        y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
        ys.append(y)
    y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
    
    y = y + u * D

    return y



# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Lambda = torch.view_as_complex(Lambda)

    Identity = torch.ones(Lambda.shape[0], device=Lambda.device)
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde

    Lambda_bar = torch.view_as_real(Lambda_bar)
    B_bar = torch.view_as_real(B_bar)

    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    # Identity = torch.ones(Lambda.shape[0], device=Lambda.device) # (replaced by -1)
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - 1))[..., None] * B_tilde
    return Lambda_bar, B_bar


def as_complex(t: torch.Tensor, dtype=torch.complex64):
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt


Initialization = Literal["dense_columns", "dense", "factorized"]


class S5SSM(torch.nn.Module):
    def __init__(
        self,
        lambdaInit: torch.Tensor,
        V: torch.Tensor,
        Vinv: torch.Tensor,
        h: int,
        p: int,
        dt_min: float,
        dt_max: float,
        liquid: bool = False,
        factor_rank: Optional[int] = None,
        discretization: Literal["zoh", "bilinear"] = "bilinear",
        bcInit: Initialization = "factorized",
        degree: int = 1,
        bidir: bool = False,
        step_scale: float = 1.0,
        bandlimit: Optional[float] = None,
    ):
        """The S5 SSM
        Args:
            lambdaInit  (complex64): Initial diagonal state matrix       (P,)
            V           (complex64): Eigenvectors used for init          (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init  (P,P)
            h           (int32):     Number of features of input seq
            p           (int32):     state size
            k           (int32):     rank of low-rank factorization (if used)
            bcInit      (string):    Specifies How B and C are initialized
                        Options: [factorized: low-rank factorization,
                                dense: dense matrix drawn from Lecun_normal]
                                dense_columns: dense matrix where the columns
                                of B and the rows of C are each drawn from Lecun_normal
                                separately (i.e. different fan-in then the dense option).
                                We found this initialization to be helpful for Pathx.
            discretization: (string) Specifies discretization method
                            options: [zoh: zero-order hold method,
                                    bilinear: bilinear transform]
            liquid:         (bool): use liquid_ssm from LiquidS4
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_scale:  (float32): allows for changing the step size, e.g. after training
                                    on a different resolution for the speech commands benchmark
        """
        super().__init__()
        self.Lambda = torch.nn.Parameter(torch.view_as_real(lambdaInit))
        self.degree = degree
        self.liquid = liquid
        self.bcInit = bcInit
        self.bidir = bidir
        self.bandlimit = bandlimit

        cp = p
        if self.bidir:
            cp *= 2

        match bcInit:
            case "complex_normal":
                self.C = torch.nn.Parameter(
                    torch.normal(0, 0.5**0.5, (h, cp), dtype=torch.complex64)
                )
                self.B = torch.nn.Parameter(
                    init_VinvB(lecun_normal(), Vinv)((p, h), torch.float)
                )
            case "dense_columns" | "dense":
                if bcInit == "dense_columns":
                    B_eigen_init = init_columnwise_VinvB
                    B_init = init_columnwise_B
                    C_init = init_rowwise_C
                elif bcInit == "dense":
                    B_eigen_init = init_VinvB
                    B_init = C_init = lecun_normal()
                # TODO: make init_*VinvB all a the same interface
                self.B = torch.nn.Parameter(
                    B_eigen_init(B_init, Vinv)((p, h), torch.float)
                )
                if self.bidir:
                    C = torch.cat(
                        [init_CV(C_init, (h, p), V), init_CV(C_init, (h, p), V)],
                        axis=-1,
                    )
                else:
                    C = init_CV(C_init, (h, p), V)
                self.C = torch.nn.Parameter(torch.view_as_real(C))
            case _:
                raise NotImplementedError(f"BC_init method {bcInit} not implemented")

        # Initialize feedthrough (D) matrix
        self.D = torch.nn.Parameter(
            torch.rand(
                h,
            )
        )
        self.log_step = torch.nn.Parameter(init_log_steps(p, dt_min, dt_max))
        match discretization:
            case "zoh":
                self.discretize = discretize_zoh
            case "bilinear":
                self.discretize = discretize_bilinear
            case _:
                raise ValueError(f"Unknown discretization {discretization}")

        if self.bandlimit is not None:
            step = step_scale * torch.exp(self.log_step)

            freqs = step / step_scale * self.Lambda[:, 1].abs() / (2 * math.pi)
            mask = torch.where(freqs < bandlimit * 0.5, 1, 0)  # (64, )
            self.C = torch.nn.Parameter(
                torch.view_as_real(torch.view_as_complex(self.C) * mask)
            )

    def initial_state(self, batch_size: Optional[int]):
        batch_shape = (batch_size,) if batch_size is not None else ()
        _, C_tilde = self.get_BC_tilde()

        return torch.zeros((*batch_shape, C_tilde.shape[-2]))

    def get_BC_tilde(self):
        match self.bcInit:
            case "dense_columns" | "dense" | "complex_normal":
                B_tilde = as_complex(self.B)
                C_tilde = self.C
            case "factorized":
                B_tilde = self.BP @ self.BH.T
                C_tilde = self.CH.T @ self.CP
        return B_tilde, C_tilde

    def forward_rnn(self, signal, prev_state, step_scale: float | torch.Tensor = 1.0):
        assert not self.bidir, "Can't use bidirectional when manually stepping"
        B_tilde, C_tilde = self.get_BC_tilde()
        step = step_scale * torch.exp(self.log_step)
        Lambda_bar, B_bar = self.discretize(self.Lambda, B_tilde, step)
        if self.degree != 1:
            assert (
                B_bar.shape[-2] == B_bar.shape[-1]
            ), "higher-order input operators must be full-rank"
            B_bar **= self.degree

        if not torch.is_tensor(step_scale) or step_scale.ndim == 0:
            step_scale = torch.ones(signal.shape[-2], device=signal.device) * step_scale
        step = step_scale[:, None] * torch.exp(self.log_step)
        # https://arxiv.org/abs/2209.12951v1, Eq. 9
        Bu = B_bar @ signal
        if self.liquid:
            Lambda_bar += Bu
        # https://arxiv.org/abs/2208.04933v2, Eq. 2
        x = Lambda_bar * prev_state + Bu
        y = (C_tilde @ x + self.D * signal).real
        return y, x

    # NOTE: can only be used as RNN OR S5(MIMO) (no mixing)
    def forward(self, signal, prev_state, step_scale: float | torch.Tensor = 1.0):
        B_tilde, C_tilde = self.get_BC_tilde()
        if self.degree != 1:
            assert (
                B_bar.shape[-2] == B_bar.shape[-1]
            ), "higher-order input operators must be full-rank"
            B_bar **= self.degree

        if not torch.is_tensor(step_scale) or step_scale.ndim == 0:
            # step_scale = torch.ones(signal.shape[-2], device=signal.device) * step_scale
            step = step_scale * torch.exp(self.log_step)
        else:
            # TODO: This is very expensive due to individual steps being multiplied by B_tilde in self.discretize
            step = step_scale[:, None] * torch.exp(self.log_step)

        Lambda_bars, B_bars = self.discretize(self.Lambda, B_tilde, step)
        # Lambda_bars, B_bars = torch.vmap(self.discretize, (None, None, 0))(self.Lambda, B_tilde, step)
        forward = apply_ssm_liquid if self.liquid else apply_ssm
        # return forward(
        #     Lambda_bars, B_bars, C_tilde, self.D, signal, prev_state, bidir=self.bidir
        # )
        forword = selective_scan
        return forword(signal, prev_state, Lambda_bars, B_bars, C_tilde, self.D)


class S5(torch.nn.Module):
    def __init__(
        self,
        width: int,
        state_width: Optional[int] = None,
        factor_rank: Optional[int] = None,
        block_count: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        liquid: bool = False,
        degree: int = 1,
        bidir: bool = False,
        bcInit: Optional[Initialization] = None,
        bandlimit: Optional[float] = None,
    ):
        super().__init__()
        state_width = state_width or width
        assert (
            state_width % block_count == 0
        ), "block_count should be a factor of state_width"

        block_size = state_width // block_count
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
        Vinv = V.conj().T
        Lambda, B, V, B_orig, Vinv = map(
            lambda v: torch.tensor(v, dtype=torch.complex64),
            (Lambda, B, V, B_orig, Vinv),
        )
        if block_count > 1:
            Lambda = Lambda[:block_size]
            V = V[:, :block_size]
            Lambda = (Lambda * torch.ones((block_count, block_size))).ravel()
            V = torch.block_diag(*([V] * block_count))
            Vinv = torch.block_diag(*([Vinv] * block_count))

        assert bool(factor_rank) != bool(
            bcInit != "factorized"
        ), "Can't have `bcInit != factorized` and `factor_rank` defined"
        bc_init = "factorized" if factor_rank is not None else (bcInit or "dense")
        self.width = width
        self.seq = S5SSM(
            Lambda,
            V,
            Vinv,
            width,
            state_width,
            dt_min,
            dt_max,
            factor_rank=factor_rank,
            bcInit=bc_init,
            liquid=liquid,
            degree=degree,
            bidir=bidir,
            bandlimit=bandlimit,
        )

    def initial_state(self, batch_size: Optional[int] = None):
        return self.seq.initial_state(batch_size)

    def forward(self, signal, prev_state, step_scale: float | torch.Tensor = 1.0):
        # NOTE: step_scale can be float | Tensor[batch] | Tensor[batch, seq]
        if not torch.is_tensor(step_scale):
            # Duplicate across batchdim
            step_scale = torch.ones(signal.shape[0], device=signal.device) * step_scale

        return torch.vmap(lambda s, ps, ss: self.seq(s, prev_state=ps, step_scale=ss))(
            signal, prev_state, step_scale
        )


class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
    
from einops import rearrange, repeat, einsum
# # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

class S5Block(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        state_dim: int,
        bidir: bool,
        block_count: int = 1,
        liquid: bool = False,
        degree: int = 1,
        factor_rank: int | None = None,
        bcInit: Optional[Initialization] = None,
        ff_mult: float = 1.0,
        glu: bool = True,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        bandlimit: Optional[float] = None,
    ):
        super().__init__()
        # self.s5 = S5(
        #     dim,
        #     state_width=state_dim,
        #     bidir=bidir,
        #     block_count=block_count,
        #     liquid=liquid,
        #     degree=degree,
        #     factor_rank=factor_rank,
        #     bcInit=bcInit,
        #     bandlimit=bandlimit,
        # )
        self.s5 = SS2D(
            d_model=dim,
            d_state=16,
    )
        self.attn_norm = torch.nn.LayerNorm(dim)
        self.attn_dropout = torch.nn.Dropout(p=attn_dropout)
        self.geglu = GEGLU() if glu else None
        self.ff_enc = torch.nn.Linear(dim, int(dim * ff_mult) * (1 + glu), bias=False)
        self.ff_dec = torch.nn.Linear(int(dim * ff_mult), dim, bias=False)
        self.ff_norm = torch.nn.LayerNorm(dim)
        self.ff_dropout = torch.nn.Dropout(p=ff_dropout)
    
    # def forward(self, x, states):
    def forward(self, x):
        # Standard transfomer-style block with GEGLU/Pre-LayerNorm
        fx = self.attn_norm(x)
        res = fx.clone()
        # x, new_state = self.s5(fx, states)
        x = self.s5(fx)
        x = F.gelu(x) + res
        x = self.attn_dropout(x)

        fx = self.ff_norm(x)
        res = fx.clone()
        x = self.ff_enc(fx)
        if self.geglu is not None:
            x = self.geglu(x)
        x = self.ff_dec(x) + res
        x = self.ff_dropout(
            x
        )  # TODO: test if should be placed inbetween ff or after ff
        # return x, new_state
        return x

# #! SS2D
# class VSSBlock(torch.nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             expand: float = 2.,
#             is_light_sr: bool = False,
#             k_bits=32,
#             **kwargs,
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,
#                                    dropout=attn_drop_rate, k_bits=k_bits, **kwargs)
#         self.drop_path = DropPath(drop_path)
#         self.skip_scale= torch.nn.Parameter(torch.ones(hidden_dim))
#         self.conv_blk = CAB(hidden_dim,is_light_sr, k_bits=k_bits)
#         self.ln_2 = torch.nn.LayerNorm(hidden_dim)
#         self.skip_scale2 = torch.nn.Parameter(torch.ones(hidden_dim))

#     def forward(self, input, x_size):
#         # x [B,HW,C]
#         B, L, C = input.shape
#         input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
#         x = self.ln_1(input)
#         x = self.self_attention(x)
#         x = input*self.skip_scale + self.drop_path(x)
#         x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
#         x = x.view(B, -1, C).contiguous()
#         return x
import selective_scan_cuda
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

class SS2D(torch.nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = torch.nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = torch.nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = torch.nn.SiLU()

        self.x_proj = (
            torch.nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            torch.nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            torch.nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            torch.nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = torch.nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = torch.nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = torch.nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = torch.nn.LayerNorm(self.d_inner)
        self.out_proj = torch.nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = torch.nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            torch.nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            torch.nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = torch.nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    
    def initial_state(self, batch_size: Optional[int]):
        batch_shape = (batch_size,) if batch_size is not None else ()
        _, C_tilde = self.get_BC_tilde()
        return torch.zeros((*batch_shape, C_tilde.shape[-2]))
    
    # def selective_scan_2d_pytorch(u, delta, A, B, C, D, x_init=None):
    #     """
    #     Args:
    #         u: (B, L, D)
    #         delta: (B, L, D)
    #         A: (L, N)
    #         B, C: (B, H, N, D)
    #         D: (L,)
    #         x_init: optional (B, H, N, D)
    #     Returns:
    #         y: (B, L, D)
    #     """
    #     u = u.permute(0, 2, 1).contiguous()
    #     delta = delta.permute(0, 2, 1).contiguous()
    #     B_, L, D_ = u.shape
    #     _, H, N, _ = B.shape
    #     device = u.device
    #     dtype = u.dtype

    #     u = u.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L, D) for broadcasting
    #     delta = delta.unsqueeze(1).unsqueeze(2)

    #     # prepare states
    #     x = torch.zeros((B_, H, N, D_), device=device, dtype=dtype) if x_init is None else x_init
    #     y_out = []

    #     for l in range(L):
    #         # extract current A, D (broadcast over B/H/N/D)
    #         A_l = A[l]  # (N,)
    #         D_l = D[l]  # scalar

    #         # recurrent update
    #         Bu = B[:, :, :, :] * u[:, :, :, l]  # (B, H, N, D)
    #         Cu = C[:, :, :, :] * u[:, :, :, l]  # (B, H, N, D)
    #         x = (1 - delta[:, :, :, l]) * x + delta[:, :, :, l] * (A_l.view(1, 1, N, 1) * x + Bu)
    #         y = x + Cu + D_l * u[:, :, :, l]
    #         y_out.append(y.squeeze(1).squeeze(1))  # drop head/N dims if not needed

    #     return torch.stack(y_out, dim=1)  # (B, L, D)


    def selective_scan_minimal_2d(self, u, delta, A, B, C, D):
        """
        Args:
            u:     (B, L, D)
            delta: (B, L, D)
            A:     (L, N)
            B:     (B, H, N, D)
            C:     (B, H, N, D)
            D:     (L,)
        Returns:
            y:     (B, L, D)
        """
        u = u.permute(0, 2, 1).contiguous()
        delta = delta.permute(0, 2, 1).contiguous()
        B_, L, D_ = u.shape
        _, H, N, _ = B.shape

        # Discretize A: shape (L, N) → broadcast to (B, H, N, D)
        deltaA = torch.exp(
            delta.transpose(1, 2).unsqueeze(1) @ A.transpose(0, 1)
        ).transpose(1, 2)  # shape: (B, D, L, N) → (B, L, D, N)
        deltaA = deltaA.permute(0, 3, 2, 1)  # (B, N, D, L) → align for scan
        deltaA = deltaA.permute(0, 2, 1, 3).contiguous()  # (B, D, N, L)

        # Discretized B * u
        deltaB_u = delta.unsqueeze(1) * B * u.unsqueeze(1).unsqueeze(2)  # (B, H, N, D)

        # initialize state
        x = torch.zeros((B_, H, N, D_), device=u.device, dtype=u.dtype)
        ys = []

        for i in range(L):
            # deltaA: (B, D, N, L) → current step
            deltaA_i = deltaA[:, :, :, i].permute(0, 3, 1, 2)  # (B, H, N, D)
            deltaB_ui = deltaB_u[..., i]  # (B, H, N, D)
            x = deltaA_i * x + deltaB_ui
            y = (x * C[..., i]).sum(dim=2)  # sum over N → (B, H, D)
            ys.append(y)

        # stack over time and reshape
        y = torch.stack(ys, dim=1)  # (B, L, H, D)
        y = y.sum(dim=2)  # sum over heads H → (B, L, D)

        # residual connection
        y = y + u * D.view(1, 1, -1)
        return y

    def selective_scan_minimal_2d(self, u, delta, A, B, C, D):
        """
        Args:
            u:     (B, L, D)
            delta: (B, L, D)
            A:     (L, N)
            B:     (B, H, N, D)
            C:     (B, H, N, D)
            D:     (L,)
        Returns:
            y:     (B, L, D)
        """
        u = u.permute(0, 2, 1).contiguous()
        delta = delta.permute(0, 2, 1).contiguous()
        B_, C_, T = u.shape
        _, H, N, _ = B.shape
        device = u.device
        dtype = u.dtype

        # Expand A, D
        A = A.view(1, C_, N)         # (1, C, N)
        D = D.view(1, C_, 1)         # (1, C, 1)

        # Expand B, C to (B, C, N, T)
        B = B.mean(dim=1)            # (B, N, T)
        C = C.mean(dim=1)            # (B, N, T)
        B = B.unsqueeze(1).expand(-1, C_, -1, -1)  # (B, C, N, T)
        C = C.unsqueeze(1).expand(-1, C_, -1, -1)  # (B, C, N, T)

        # Init state x
        x = torch.zeros((B_, C_, N), device=device, dtype=dtype)

        y_out = []

        for t in range(T):
            u_t = u[:, :, t].unsqueeze(-1)          # (B, C, 1)
            delta_t = delta[:, :, t].unsqueeze(-1)  # (B, C, 1)

            A_x = A * x                              # (B, C, N)
            Bu = B[:, :, :, t] * u_t                # (B, C, N)

            x = (1 - delta_t) * x + delta_t * (A_x + Bu)

            Cu = C[:, :, :, t] * u_t                # (B, C, N)
            y_t = torch.sum(Cu + x, dim=2) + D.squeeze(-1) * u[:, :, t]  # (B, C)

            y_out.append(y_t)

        y = torch.stack(y_out, dim=2)  # (B, C, T)
        return y


    def selective_scan_minimal(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        u = u.permute(0, 2, 1).contiguous()
        delta = delta.permute(0, 2, 1).contiguous()
        # import pdb;pdb.set_trace()
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = torch.nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def forward_core(self, x: torch.Tensor):
        # import pdb;pdb.set_trace()
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        # out_y = self.selective_scan(
        #     xs, dts,
        #     As, Bs, Cs, Ds, z=None,
        #     delta_bias=dt_projs_bias,
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # import pdb;pdb.set_trace()
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        # import pdb; pdb.set_trace()
        # B, Hw, L, C = x.shape
        # x = x.permute(0, 3, 1, 2).contiguous()
        # import pdb;pdb.set_trace()
        B, H, W, C = x.shape
        # print(x.shape)
        # import pdb;pdb.set_trace()
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class ChannelAttention(torch.nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=1):
        super(ChannelAttention, self).__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            torch.nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(torch.nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=1):
        super(CAB, self).__init__()
        if is_light_sr: # we use dilated-conv & DWConv for lightSR for a large ERF
            compress_ratio = 2 
            self.cab = torch.nn.Sequential(
                torch.nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 0),
                torch.nn.Conv2d(num_feat//compress_ratio, num_feat // compress_ratio, 3, 1, 1,groups=num_feat//compress_ratio),
                torch.nn.GELU(),
                torch.nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 0),
                torch.nn.Conv2d(num_feat, num_feat, 3,1,padding=2,groups=num_feat,dilation=2),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else:
            self.cab = torch.nn.Sequential(
                torch.nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                torch.nn.GELU(),
                torch.nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        # import pdb; pdb.set_trace()
        return self.cab(x)

if __name__ == "__main__":
    import lovely_tensors as lt

    lt.monkey_patch()

    def tensor_stats(t: torch.Tensor):  # Clone of lovely_tensors for complex support
        return f"tensor[{t.shape}] n={t.shape.numel()}, u={t.mean()}, s={round(t.std().item(), 3)} var={round(t.var().item(), 3)}\n"

    x = torch.rand([2, 256, 32]).cuda()
    model = S5(32, 32, factor_rank=None).cuda()
    print("B", tensor_stats(model.seq.B.data))
    print("C", tensor_stats(model.seq.C.data))
    # print('B', tensor_stats(model.seq.BH.data), tensor_stats(model.seq.BP.data))
    # print('C', tensor_stats(model.seq.CH.data), tensor_stats(model.seq.CP.data))
    # FIXME: unstable initialization
    # state = model.initial_state(256)
    # res = model(x, prev_state=state)
    # print(res.shape, res.dtype, res)
    res = model(x)  # warm-up
    print(res.shape, res.dtype, res)

    # Example 2: (B, L, H) inputs
    x = torch.rand([2, 256, 32]).cuda()
    model = S5Block(32, 32, False).cuda()
    res = model(x)
    print(res.shape, res.dtype, res)

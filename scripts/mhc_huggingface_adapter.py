import torch
import torch.nn as nn
from typing import Tuple, Optional
import types
import math

class SinkhornKnopp(nn.Module):
    def __init__(self, num_iters: int = 20):
        super().__init__()
        self.num_iters = num_iters
        
    def forward(self, log_alpha: torch.Tensor) -> torch.Tensor:
        # Paper Sec 4.3.1: "traverses the entire iteration" in backward
        # By using native PyTorch ops, autograd handles this automatically.
        P = log_alpha.to(torch.float32)
        for _ in range(self.num_iters):
            # Row normalization
            P = P - torch.logsumexp(P, dim=-1, keepdim=True)
            # Column normalization
            P = P - torch.logsumexp(P, dim=-2, keepdim=True)
        return torch.exp(P).to(log_alpha.dtype)

class ManualRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-20): # Restored to 1e-20 as per Paper
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        x_f32 = x.to(torch.float32)
        norm_x = x_f32.pow(2).mean(-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(norm_x + self.eps)
        return (self.weight * x_normed).to(x.dtype)

def sanity_check(tensor, name, layer_idx=None, step=None):
    if not torch.isfinite(tensor).all():
        err_msg = f"FATAL ERROR: Non-finite values detected in {name}"
        if layer_idx is not None: err_msg += f" (Layer {layer_idx})"
        if step is not None: err_msg += f" (Step {step})"
        # Log stats
        stats = f"Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}, Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}"
        print(f"\n{err_msg}\n{stats}")
        raise ValueError(err_msg)

class DynamicMHCAdapter(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, init_identity: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.nC = num_heads * hidden_size # Paper Section 4.2
        self.output_dim = (num_heads * num_heads) + num_heads + num_heads
        
        # Linear projections for dynamic mappings (Paper Eq 7)
        self.dynamic_proj = nn.Linear(self.nC, self.output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        
        # Learnable gating factors (Paper Appendix Table 5: α = 0.01)
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        
        self.rms_norm = ManualRMSNorm(self.nC, eps=1e-20) # Flattened context norm
        self.sinkhorn = SinkhornKnopp(num_iters=20)
        
        if init_identity:
            self.initialize_parameters()

    def initialize_parameters(self):
        # Paper Sec 3.1 & 4.1: Restore identity mapping property
        # H_res -> I, H_pre -> 1/n, H_post -> 1.0 (Initial Forward Preservation)
        with torch.no_grad():
            self.alpha_res.fill_(0.01)
            self.alpha_pre.fill_(0.01)
            self.alpha_post.fill_(0.01)
            
            # Start with ZERO contribution from the dynamic part to ensure
            # a perfectly stable identity mapping at Step 0.
            self.dynamic_proj.weight.zero_()
            
            n = self.num_heads
            # b_res: Diagonal dominant for Identity (Log-space)
            res_bias = torch.zeros(n, n)
            res_bias.fill_(-10.0) # Suppress off-diagonal
            res_bias.diagonal().fill_(10.0) # Favor diagonal
            self.bias.data[:n*n] = res_bias.flatten()
            
            # b_pre: sigma(b) = 1/n = 0.25 (for n=4) -> logit(0.25) approx -1.1
            self.bias.data[n*n : n*n+n].fill_(-1.1)
            
            # b_post: 2*sigma(b) = 1.0 -> sigma(b) = 0.5 -> logit(0.5) = 0
            self.bias.data[n*n+n :].fill_(0.0)

    def forward(self, x_expanded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, n, C = x_expanded.shape
        x_flat = x_expanded.view(B, L, n * C)
        
        # Paper Eq 15: r = ||x||_2 / sqrt(nC)
        x_f32 = x_flat.to(torch.float32)
        r = torch.norm(x_f32, p=2, dim=-1, keepdim=True) / (self.nC**0.5) + self.rms_norm.eps
        
        # Paper Eq 14: project first
        proj = self.dynamic_proj(x_f32) # (B, L, n2+2n)
        
        # Paper Eq 16: H_tilde = (alpha * proj) / r + bias
        n2 = n * n
        l_res = (self.alpha_res * proj[..., :n2]) / r + self.bias[:n2]
        l_pre = (self.alpha_pre * proj[..., n2 : n2+n]) / r + self.bias[n2 : n2+n]
        l_post = (self.alpha_post * proj[..., n2+n :]) / r + self.bias[n2+n :]
        
        # Gating Manifolds (Eq 8 & 17-18)
        H_res = self.sinkhorn(l_res.reshape(B, L, n, n))
        H_pre = torch.sigmoid(l_pre)
        H_post = 2.0 * torch.sigmoid(l_post)
        
        # Sanity check tensors
        idx = getattr(self, "layer_idx", -1)
        # sanity_check(H_res, "H_res", idx)
        # sanity_check(H_pre, "H_pre", idx)
        # sanity_check(H_post, "H_post", idx)
            
        return H_res, H_pre, H_post

def check_weights(model, step=None):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not torch.isfinite(param).all():
                stats = f"Mean: {param.mean().item():.4f}, Max: {param.max().item():.4f}"
                print(f"FATAL: Weight {name} exploded at step {step}. {stats}")
                raise ValueError(f"Exploding weights in {name}")
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print(f"FATAL: Gradient for {name} is NaN/Inf at step {step}")
                raise ValueError(f"NaN gradient in {name}")

def freeze_backbone_but_mhc(model):
    """
    Freezes all parameters in the model except those in MHC adapters.
    """
    print("Freezing backbone parameters...")
    for name, param in model.named_parameters():
        if "mhc_adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Verify
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Frozen backbone. Trainable parameters: {len(trainable)} tensors (MHC adapters).")

def backward_hook(module, grad_input, grad_output):
    if hasattr(module, "name"):
        name = module.name
    else:
        name = module.__class__.__name__
    
    # check grad_input
    if grad_input is not None:
        for i, g in enumerate(grad_input):
            if g is not None and not torch.isfinite(g).all():
                print(f"FATAL: NaN gradient detected in INPUT {i} of {name}")
                raise ValueError(f"NaN gradient input in {name}")
                
    # check grad_output
    if grad_output is not None:
        for i, g in enumerate(grad_output):
            if g is not None and not torch.isfinite(g).all():
                print(f"FATAL: NaN gradient detected in OUTPUT {i} of {name}")
                raise ValueError(f"NaN gradient output in {name}")

def patch_gpt_oss_with_mhc(model, num_heads: int = 4, freeze_backbone: bool = False):
    import torch
    
    # Discovery
    layers = None
    for name, module in model.named_modules():
        if (name.endswith(".layers") or name.endswith(".h")) and isinstance(module, torch.nn.ModuleList):
            if len(module) > 20: 
                layers = module
                break
    
    if layers is None: raise AttributeError("Could not find layers ModuleList")
        
    num_layers = len(layers)
    hidden_size = model.config.hidden_size
    
    for i in range(num_layers):
        layer = layers[i]
        device = next(layer.parameters()).device
        dtype = next(layer.parameters()).dtype
        
        adapter = DynamicMHCAdapter(hidden_size=hidden_size, num_heads=num_heads, init_identity=True).to(device)
        adapter.layer_idx = i
        layer.mhc_adapter = adapter
        layer.original_forward = layer.forward
        
        # Register Hooks for Debugging (Step 3 Crash)
        adapter.flux_name = f"Layer_{i}_Adapter"
        adapter.register_full_backward_hook(backward_hook)
        adapter.sinkhorn.name = f"Layer_{i}_Sinkhorn"
        adapter.sinkhorn.register_full_backward_hook(backward_hook)
        adapter.dynamic_proj.name = f"Layer_{i}_Proj"
        adapter.dynamic_proj.register_full_backward_hook(backward_hook)
        
        def make_mhc_forward(l, idx):
            def mhc_forward(self, hidden_states, *args, **kwargs):
                # 1. First layer expansion (Identity preservation)
                if len(hidden_states.shape) == 3:
                    # Paper Sec 3.1: Expand single-stream to n-streams
                    # Initializing with copies ensures x_in = sum(1/n * x) = x
                    hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.mhc_adapter.num_heads, 1)
                
                sanity_check(hidden_states, "input_hidden_states", idx)
                
                # 2. Get Manifold-Constrained Matrices
                H_res, H_pre, H_post = self.mhc_adapter(hidden_states)
                
                # Cast for mixing
                H_res = H_res.to(hidden_states.dtype)
                H_pre = H_pre.to(hidden_states.dtype)
                H_post = H_post.to(hidden_states.dtype)
                
                # 3. Mixing and Residual Branch (Paper Eq 3)
                # x_mixed = H_res @ x_l
                x_mixed = torch.einsum('...ij, ...jk -> ...ik', H_res, hidden_states)
                sanity_check(x_mixed, "x_mixed", idx)
                
                # x_in = H_pre * x_l
                x_in = torch.einsum('...j, ...jk -> ...k', H_pre, hidden_states)
                sanity_check(x_in, "x_in", idx)
                
                # 4. Layer Function F(.)
                outputs = self.original_forward(x_in, *args, **kwargs)
                x_out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                # NO CLAMPING (User request) - Relying on paper-strict stability
                sanity_check(x_out, "x_out", idx)
                
                # F(x) = (residual+F(x)) - x
                delta = x_out - x_in
                sanity_check(delta, "delta", idx)
                
                # 5. Update: H_post^T * F(x)
                update = torch.einsum('...i, ...k -> ...ik', H_post, delta)
                sanity_check(update, "update", idx)
                
                x_next = x_mixed + update
                sanity_check(x_next, "x_next", idx)
                
                # Collapsing at final layer
                if idx == num_layers - 1:
                    # Use mean to preserve signal magnitude ~1.0
                    x_next = x_next.mean(dim=2)
                    sanity_check(x_next, "x_final_collapsed", idx)
                
                if isinstance(outputs, (tuple, list)):
                    return (x_next,) + outputs[1:]
                return x_next
            return mhc_forward
            
        layer.forward = types.MethodType(make_mhc_forward(layer, i), layer)

    print(f"Patched {num_layers} layers with Paper-Strict Dynamic MHC adapters (Sanity Checks enabled).")
    if freeze_backbone:
        freeze_backbone_but_mhc(model)
    return model

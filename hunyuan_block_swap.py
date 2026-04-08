"""
HunyuanImage-3.0 Block Swap Manager - V2 Unified Node Support

Kijai-style explicit block swapping for transformer blocks.
Moves blocks between GPU and CPU during forward pass to enable
large generations on limited VRAM.

Uses pre-allocated pinned CPU buffers to eliminate CRT heap fragmentation
on Windows. Without this, each block.to('cpu') allocates ~2GB via
_aligned_malloc which the CRT never returns to the OS, causing RSS to
grow monotonically across generations.

Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BlockSwapConfig:
    """Configuration for block swapping."""
    blocks_to_swap: int = 0           # 0 = all on GPU, no swapping
    prefetch_blocks: int = 1          # Number of blocks to prefetch async
    use_non_blocking: bool = True     # Use non-blocking transfers
    offload_device: str = "cpu"       # Target device for offloaded blocks ("cpu" or "cuda:1")
    swap_start_idx: int = 0           # Start swapping from this block index
    debug: bool = False               # Enable verbose debug logging
    
    def __post_init__(self):
        if self.blocks_to_swap < 0:
            raise ValueError("blocks_to_swap must be >= 0")
        if self.prefetch_blocks < 0:
            raise ValueError("prefetch_blocks must be >= 0")


@dataclass
class BlockSwapStats:
    """Statistics from block swap operations."""
    total_swaps_to_gpu: int = 0
    total_swaps_to_cpu: int = 0
    total_swap_time_seconds: float = 0.0
    blocks_currently_on_gpu: int = 0
    blocks_currently_on_cpu: int = 0
    
    def __str__(self) -> str:
        return (
            f"BlockSwapStats(gpu={self.blocks_currently_on_gpu}, "
            f"cpu={self.blocks_currently_on_cpu}, "
            f"swaps={self.total_swaps_to_gpu + self.total_swaps_to_cpu}, "
            f"time={self.total_swap_time_seconds:.2f}s)"
        )


class BlockSwapManager:
    """
    Manages transformer block swapping between GPU and CPU.
    
    This implements Kijai-style explicit block swapping where:
    1. Some blocks stay on GPU (always resident)
    2. Other blocks are swapped in/out during forward pass
    3. Prefetching loads next block while current one runs
    
    Memory Management:
        Pre-allocates pinned CPU buffers for all swappable blocks during
        setup_initial_placement(). All subsequent GPU<->CPU transfers reuse
        these buffers - zero new CPU allocations per generation. This prevents
        Windows CRT heap fragmentation that causes RSS to grow unboundedly.
        Pinned memory also enables faster async DMA for prefetch transfers.
    
    Usage:
        config = BlockSwapConfig(blocks_to_swap=10)
        manager = BlockSwapManager(model, config)
        manager.setup_initial_placement()
        manager.install_hooks()  # Enable automatic swapping during forward pass
        
        # Generation will now automatically swap blocks
        output = model.generate_image(...)
        
        manager.remove_hooks()  # Clean up when done
    """
    
    def __init__(
        self,
        model: Any,
        config: BlockSwapConfig,
        target_device: str = "cuda:0"
    ):
        """
        Initialize block swap manager.
        
        Args:
            model: The Hunyuan model with transformer blocks
            config: Block swap configuration
            target_device: Primary GPU device for computation
        """
        self.model = model
        self.config = config
        self.target_device = torch.device(target_device)
        self.offload_device = torch.device(config.offload_device)
        
        # Find transformer blocks
        self.blocks = self._find_transformer_blocks()
        self.num_blocks = len(self.blocks)
        
        # Track where each block is located
        self.block_locations: Dict[int, torch.device] = {}
        
        # Track installed hooks for removal
        self._hook_handles: List[Any] = []
        self._hooks_installed: bool = False
        
        # Stats
        self.stats = BlockSwapStats()
        
        # CUDA streams for async transfers
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._prefetch_events: Dict[int, torch.cuda.Event] = {}
        
        # Memory per block (calculated once)
        self._bytes_per_block: Optional[int] = None
        
        # Check if model uses INT8 quantization (bitsandbytes)
        self._has_int8_layers = self._detect_int8_layers()
        
        # Check if model uses NF4 quantization (bitsandbytes Params4bit)
        self._has_nf4_layers = self._detect_nf4_layers()
        
        # Pre-allocated pinned CPU buffers to prevent CRT heap fragmentation.
        # Maps block_idx -> {param_name: pinned_cpu_tensor, ...}
        # Created by _create_cpu_buffer_store() during setup_initial_placement().
        # NOTE: Not used for NF4 models (Params4bit has quant_state that the
        # buffered path cannot handle — we use block.to() instead).
        # NOTE: Not used for INT8 models because on Windows WDDM, pinned CPU
        # memory (cudaHostAlloc) is mapped into GPU address space and counted
        # against GPU memory by cuMemGetInfo.  67GB of pinned buffers can
        # exhaust a 96GB GPU, leaving no room for inference activations.
        self._cpu_param_store: Dict[int, Dict[str, torch.Tensor]] = {}
        
        logger.info(f"BlockSwapManager initialized: {self.num_blocks} blocks found")
        logger.info(f"  Config: swap={config.blocks_to_swap}, prefetch={config.prefetch_blocks}")
        if self._has_nf4_layers:
            logger.info(f"  NF4 quantized layers detected - using block.to() path (quant_state safe)")
        if self._has_int8_layers:
            logger.info(f"  INT8 quantized layers detected - using block.to() path "
                       f"(pinned buffers consume GPU address space on Windows WDDM)")
    
    def _detect_int8_layers(self) -> bool:
        """Check if any block contains bitsandbytes Linear8bitLt layers."""
        try:
            from bitsandbytes.nn import Linear8bitLt
            for block in self.blocks:
                for module in block.modules():
                    if isinstance(module, Linear8bitLt):
                        return True
        except ImportError:
            pass
        return False
    
    def _detect_nf4_layers(self) -> bool:
        """Check if any block contains bitsandbytes NF4/4-bit quantized parameters.
        
        NF4 parameters (Params4bit) store quantized data + quant_state (absmax,
        code, offset). The buffered block swap path only moves param.data and
        misses quant_state, leaving stale GPU pointers that cause
        cudaErrorIllegalAddress. We must use block.to() for NF4 blocks instead.
        """
        try:
            from bitsandbytes.nn import Params4bit
            for block in self.blocks:
                for param in block.parameters():
                    if isinstance(param, Params4bit):
                        return True
        except ImportError:
            pass
        # Fallback: check for quant_state attribute
        if not self.blocks:
            return False
        for param in self.blocks[0].parameters():
            if hasattr(param, 'quant_state') and param.quant_state is not None:
                return True
        return False
    
    def _fix_int8_state_devices(self, block: nn.Module, device: torch.device) -> None:
        """
        Fix bitsandbytes Linear8bitLt weight and state device after block.to().
        
        bitsandbytes has a bug where block.to(device) does NOT move weight.CB,
        weight.SCB, state.CB, or state.SCB to the target device. This is because:
        
        1. Module._apply(fn) calls fn(param) -> Int8Params.to() which creates a NEW
           Int8Params with CB/SCB on the target device, but Module._apply only copies
           param.data (the raw tensor), discarding the new CB/SCB attributes.
        2. Linear8bitLt.to() overrides .to() to handle state.CB/SCB, but this override
           is never called when a PARENT module calls .to() (Module._apply recurses
           via child._apply, not child.to()).
        
        The INT8 lifecycle:
        - Before first forward: weight.CB has the compressed buffer, state.CB is None
        - After first forward: init_8bit_state() moves weight.CB -> state.CB, weight.CB = None
        - After matmul: weight.data is set to state.CB (they share the same tensor)
        
        CRITICAL ALIASING INVARIANT:
        After init_8bit_state(), weight.data and state.CB point to the SAME tensor.
        We must maintain this alias during device transfers. If we create separate
        tensors via state.CB.to(device), each block's memory is DOUBLED — both on
        GPU (causing OOM) and in pinned CPU RAM (causing massive RSS bloat).
        Fix: always set state.CB = weight.data after moving weight.data.
        
        This method explicitly moves ALL int8 data to the target device.
        """
        if not self._has_int8_layers:
            return
        
        try:
            from bitsandbytes.nn import Linear8bitLt
        except ImportError:
            return
        
        fixed_count = 0
        for name, module in block.named_modules():
            if isinstance(module, Linear8bitLt):
                # Fix weight.CB/SCB (before first forward, these hold the compressed data)
                #
                # ALIASING FIX for weight.CB:
                # Before init_8bit_state(), weight.CB holds the compressed int8
                # buffer — the same content as weight.data.  block.to(device)
                # already moved weight.data to `device` via _apply(fn), but
                # weight.CB is an attribute NOT covered by _apply, so it stays
                # on the source device.  Creating a SEPARATE GPU tensor via
                # weight.CB.to(device) DOUBLES the memory per block.  Instead,
                # alias weight.CB = weight.data when shapes match.
                weight = module.weight
                if hasattr(weight, 'CB') and weight.CB is not None and weight.CB.device != device:
                    if (weight.data.device == device
                            and weight.data.shape == weight.CB.shape
                            and weight.data.dtype == weight.CB.dtype):
                        # Alias — zero-copy, prevents doubling
                        weight.CB = weight.data
                        fixed_count += 1
                        if self.config.debug or not hasattr(self, '_int8_fix_logged'):
                            logger.info(f"  [INT8 fix] {name}: weight.CB aliased to weight.data on {device}")
                    else:
                        # Shape/dtype mismatch (e.g. before quantization) — must copy
                        if self.config.debug or not hasattr(self, '_int8_fix_logged'):
                            logger.info(f"  [INT8 fix] {name}: weight.CB {weight.CB.device} -> {device}")
                        weight.CB = weight.CB.to(device)
                        fixed_count += 1
                if hasattr(weight, 'SCB') and weight.SCB is not None and weight.SCB.device != device:
                    if self.config.debug or not hasattr(self, '_int8_fix_logged'):
                        logger.info(f"  [INT8 fix] {name}: weight.SCB {weight.SCB.device} -> {device}")
                    weight.SCB = weight.SCB.to(device)
                    fixed_count += 1
                
                # Fix state.CB/SCB (after first forward, these hold the compressed data)
                #
                # ALIASING FIX: After init_8bit_state(), state.CB and weight.data
                # are the SAME tensor. The main param loop (or block.to()) has
                # already moved weight.data to `device`. Re-alias state.CB to
                # weight.data instead of creating a SEPARATE tensor via
                # state.CB.to(device) — that would double the memory per block.
                if hasattr(module, 'state'):
                    state = module.state
                    if state.CB is not None and state.CB.device != device:
                        if weight.data.device == device:
                            # weight.data already on target — re-alias (zero-copy)
                            state.CB = weight.data
                            fixed_count += 1
                            if self.config.debug or not hasattr(self, '_int8_fix_logged'):
                                logger.info(f"  [INT8 fix] {name}: state.CB aliased to weight.data on {device}")
                        else:
                            # weight.data not yet on target (shouldn't happen in normal
                            # flow, but handle gracefully)
                            if self.config.debug or not hasattr(self, '_int8_fix_logged'):
                                logger.info(f"  [INT8 fix] {name}: state.CB {state.CB.device} -> {device}")
                            state.CB = state.CB.to(device)
                            fixed_count += 1
                    if state.SCB is not None and state.SCB.device != device:
                        # SCB (scales) is small — no significant doubling concern
                        if hasattr(weight, 'SCB') and weight.SCB is not None and weight.SCB.device == device:
                            state.SCB = weight.SCB
                        else:
                            state.SCB = state.SCB.to(device)
                        fixed_count += 1
                        if self.config.debug or not hasattr(self, '_int8_fix_logged'):
                            logger.info(f"  [INT8 fix] {name}: state.SCB -> {device}")
        
        if fixed_count > 0 and not hasattr(self, '_int8_fix_logged'):
            self._int8_fix_logged = True
            logger.info(f"  [INT8 fix] Fixed {fixed_count} INT8 tensor(s) for this block -> {device}")
    
    def _find_transformer_blocks(self) -> List[nn.Module]:
        """
        Find transformer blocks in the model.
        
        Hunyuan models have different structures, so we try multiple paths.
        """
        blocks = []
        
        # Try common paths for transformer blocks
        paths_to_try = [
            # Hunyuan-style paths
            ("model", "blocks"),
            ("model", "transformer_blocks"),
            ("model", "layers"),  # This is the correct path for HunyuanImage3
            ("transformer", "blocks"),
            ("blocks",),
            ("layers",),
            # Diffusers-style paths
            ("model", "transformer", "blocks"),
        ]
        
        for path in paths_to_try:
            obj = self.model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                
                # Check if it's a list/tuple of modules
                if isinstance(obj, (list, tuple, nn.ModuleList)):
                    blocks = list(obj)
                    logger.debug(f"Found {len(blocks)} blocks at path: {'.'.join(path)}")
                    break
            except AttributeError:
                continue
        
        if not blocks:
            # Fallback: search for anything that looks like transformer blocks
            for name, module in self.model.named_modules():
                if 'block' in name.lower() and hasattr(module, 'forward'):
                    # Check if it's a container with multiple blocks
                    if isinstance(module, (nn.ModuleList, nn.Sequential)):
                        blocks = list(module)
                        logger.debug(f"Found {len(blocks)} blocks in module: {name}")
                        break
        
        if not blocks:
            logger.warning("Could not find transformer blocks in model")
            logger.warning("Block swapping will be disabled")
        
        return blocks
    
    @property
    def bytes_per_block(self) -> int:
        """Get memory size per block in bytes."""
        if self._bytes_per_block is None and self.blocks:
            sample_block = self.blocks[0]
            self._bytes_per_block = sum(
                p.numel() * p.element_size() for p in sample_block.parameters()
            )
        return self._bytes_per_block or 0
    
    @property
    def gb_per_block(self) -> float:
        """Get memory size per block in GB."""
        return self.bytes_per_block / (1024**3)
    
    def _detect_actual_block_placement(self) -> Dict[int, torch.device]:
        """
        Detect actual block placement by checking where block parameters are.
        
        This is essential for device_map="auto" models where accelerate
        has already placed blocks across GPU/CPU based on memory limits.
        
        Returns:
            Dict mapping block index to actual device
        """
        placement = {}
        gpu_count = 0
        cpu_count = 0
        meta_count = 0
        
        for i, block in enumerate(self.blocks):
            # Check first parameter to determine block's device
            try:
                first_param = next(block.parameters())
                device = first_param.device
                placement[i] = device
                
                if device.type == 'cuda':
                    gpu_count += 1
                elif device.type == 'cpu':
                    cpu_count += 1
                elif device.type == 'meta':
                    # Meta = CPU-offloaded by accelerate's device_map="auto".
                    # Accelerate keeps weights as meta tensors and uses
                    # AlignDevicesHook to load them to the execution device
                    # on-the-fly during forward pass. This is NORMAL.
                    meta_count += 1
            except StopIteration:
                # Block has no parameters? Unusual but possible
                placement[i] = self.target_device
                gpu_count += 1
        
        if meta_count > 0:
            logger.info(f"{meta_count} blocks on 'meta' device (CPU-offloaded via accelerate hooks)")
        
        logger.info(f"Actual block placement: {gpu_count} on GPU, {cpu_count} on CPU, {meta_count} CPU-offloaded (meta)")
        
        return placement, gpu_count, cpu_count
    
    # ------------------------------------------------------------------
    # Pinned CPU buffer store - prevents CRT heap fragmentation
    # ------------------------------------------------------------------
    
    def _create_cpu_buffer_store(self) -> None:
        """Pre-allocate pinned CPU buffers for all swappable blocks.
        
        This is THE key fix for Windows CRT heap fragmentation. Without this,
        each block.to('cpu') allocates ~2GB of new CPU memory via
        _aligned_malloc. Over 30 diffusion steps x 10 blocks x 2 directions,
        that's ~1.2TB churned through the CRT heap per generation. The CRT
        never returns these pages to the OS, so RSS grows monotonically.
        
        With pinned buffers:
        - GPU->CPU: copy into pre-existing buffer - zero new allocations
        - CPU->GPU: transfer from pinned memory - uses DMA, 2-3x faster
        - Buffers are reused across all steps of all generations
        
        Called at the end of setup_initial_placement() after blocks are on CPU.
        """
        if self.config.blocks_to_swap == 0:
            return
        
        # NF4 models: skip pinned buffer creation entirely.
        # Params4bit has quant_state (absmax, code, offset) that the buffered
        # path doesn't handle. Without quant_state, the dequantize kernel
        # accesses stale GPU pointers → cudaErrorIllegalAddress.
        # Instead, we fall back to block.to() which properly handles
        # Params4bit.to() → QuantState.to() for all internal tensors.
        if self._has_nf4_layers:
            logger.info("NF4 model: skipping pinned buffer store (block.to() handles quant_state)")
            return
        
        # INT8 models: skip pinned buffer creation entirely.
        # On Windows WDDM, pinned CPU memory (cudaHostAlloc) is mapped into
        # the GPU's address space and cuMemGetInfo counts it as consumed GPU
        # memory.  With 28 blocks × ~2.4GB = ~67GB of pinned buffers, this
        # leaves only ~4GB free on a 96GB GPU → Fatal abort on first matmul.
        # Additionally, INT8 has complex aliasing between weight.data,
        # weight.CB, and state.CB that the buffered path cannot preserve
        # without doubling memory.  By falling back to block.to() +
        # _fix_int8_state_devices (with aliasing fix), both issues are
        # resolved.  The tradeoff is potential CRT heap fragmentation on
        # Windows, but INT8 users typically have ample system RAM.
        if self._has_int8_layers:
            logger.info("INT8 model: skipping pinned buffer store "
                       "(pinned memory consumes GPU address space on Windows WDDM)")
            return
        
        swap_start = self.num_blocks - self.config.blocks_to_swap
        total_bytes = 0
        pinned_count = 0
        fallback_count = 0
        
        for block_idx in range(swap_start, self.num_blocks):
            block = self.blocks[block_idx]
            store: Dict[str, torch.Tensor] = {}
            
            # Parameters (weights, biases, etc.)
            for name, param in block.named_parameters():
                buf, was_pinned = self._alloc_pinned_buffer(param.data)
                # Copy current data into the pinned buffer
                if param.device.type == 'cpu':
                    buf.copy_(param.data)
                    # Point param at the pinned buffer immediately so even
                    # the first CPU->GPU transfer benefits from DMA
                    param.data = buf
                store[name] = buf
                total_bytes += buf.numel() * buf.element_size()
                if was_pinned:
                    pinned_count += 1
                else:
                    fallback_count += 1
            
            # Buffers (registered via register_buffer, e.g. position embeddings)
            for name, buffer in block.named_buffers():
                key = f"__buffer__{name}"
                if key in store:
                    continue  # Already covered (shouldn't happen, but safety)
                buf, was_pinned = self._alloc_pinned_buffer(buffer.data)
                if buffer.device.type == 'cpu':
                    buf.copy_(buffer.data)
                    buffer.data = buf
                store[key] = buf
                total_bytes += buf.numel() * buf.element_size()
                if was_pinned:
                    pinned_count += 1
                else:
                    fallback_count += 1
            
            # INT8 bitsandbytes state tensors (CB, SCB) - not in named_parameters
            if self._has_int8_layers:
                self._create_int8_buffer_store(block, store)
            
            self._cpu_param_store[block_idx] = store
        
        num_blocks_stored = len(self._cpu_param_store)
        logger.info(
            f"Created pinned CPU buffer store: {num_blocks_stored} blocks, "
            f"{total_bytes / 1024**3:.2f} GB "
            f"({pinned_count} pinned, {fallback_count} unpinned fallback)"
        )
    
    @staticmethod
    def _alloc_pinned_buffer(reference_tensor: torch.Tensor) -> tuple:
        """Allocate a pinned CPU buffer matching the shape/dtype of reference_tensor.
        
        Returns:
            (buffer, was_pinned) - buffer is always on CPU, was_pinned indicates
            whether pin_memory succeeded.
        """
        try:
            buf = torch.empty(
                reference_tensor.shape,
                dtype=reference_tensor.dtype,
                device='cpu',
                pin_memory=True,
            )
            return buf, True
        except Exception:
            # Pinning can fail if CUDA host allocator is exhausted or
            # if CUDA isn't initialized yet. Fall back to regular CPU.
            buf = torch.empty(
                reference_tensor.shape,
                dtype=reference_tensor.dtype,
                device='cpu',
            )
            return buf, False
    
    def _create_int8_buffer_store(self, block: nn.Module, store: Dict[str, torch.Tensor]) -> None:
        """Create pinned buffers for INT8 bitsandbytes state tensors.
        
        INT8 lifecycle:
        - Before first forward: weight.CB has compressed buffer, state.CB is None
        - After first forward: init_8bit_state() moves weight.CB -> state.CB
        - We snapshot whatever exists now; any new tensors that appear later
          (after init_8bit_state) are handled lazily in _move_int8_to_cpu_buffered.
        """
        try:
            from bitsandbytes.nn import Linear8bitLt
        except ImportError:
            return
        
        for name, module in block.named_modules():
            if not isinstance(module, Linear8bitLt):
                continue
            
            weight = module.weight
            
            # weight.CB / weight.SCB (exist before first forward)
            for attr in ('CB', 'SCB'):
                tensor = getattr(weight, attr, None)
                if tensor is None:
                    continue
                key = f"__int8__{name}__weight_{attr}"
                buf, _ = self._alloc_pinned_buffer(tensor)
                if tensor.device.type == 'cpu':
                    buf.copy_(tensor)
                    setattr(weight, attr, buf)
                store[key] = buf
            
            # state.CB / state.SCB (exist after first forward)
            if hasattr(module, 'state'):
                state = module.state
                for attr in ('CB', 'SCB'):
                    tensor = getattr(state, attr, None)
                    if tensor is None:
                        continue
                    key = f"__int8__{name}__state_{attr}"
                    buf, _ = self._alloc_pinned_buffer(tensor)
                    if tensor.device.type == 'cpu':
                        buf.copy_(tensor)
                        setattr(state, attr, buf)
                    store[key] = buf
    
    def _move_int8_to_cpu_buffered(
        self,
        block: nn.Module,
        store: Dict[str, torch.Tensor],
        non_blocking: bool,
    ) -> None:
        """Move INT8 state tensors into pre-allocated pinned CPU buffers.
        
        Lazily creates pinned buffers for tensors that didn't exist at
        snapshot time (e.g. state.CB after init_8bit_state runs).
        
        ALIASING FIX: After init_8bit_state(), state.CB and weight.data are
        the SAME tensor. The main param loop in _move_block_to_device() has
        already moved weight.data into its pinned buffer. We must alias
        state.CB to that same buffer instead of allocating a separate one.
        Without this, every swapped block's memory is doubled in pinned RAM
        AND when loaded back to GPU.
        """
        try:
            from bitsandbytes.nn import Linear8bitLt
        except ImportError:
            return
        
        for name, module in block.named_modules():
            if not isinstance(module, Linear8bitLt):
                continue
            
            weight = module.weight
            
            # weight.CB / weight.SCB (exist before first forward, None after)
            for attr in ('CB', 'SCB'):
                tensor = getattr(weight, attr, None)
                if tensor is None or tensor.device.type == 'cpu':
                    continue
                key = f"__int8__{name}__weight_{attr}"
                if key in store:
                    store[key].copy_(tensor, non_blocking=non_blocking)
                    setattr(weight, attr, store[key])
                else:
                    # Lazy creation for tensors that appeared after snapshot
                    buf, _ = self._alloc_pinned_buffer(tensor)
                    buf.copy_(tensor, non_blocking=non_blocking)
                    store[key] = buf
                    setattr(weight, attr, buf)
            
            # state.CB / state.SCB (exist after first forward)
            #
            # ALIASING FIX for state.CB:
            # After init_8bit_state(), state.CB IS weight.data (same tensor).
            # The main param loop already copied weight.data into its pinned
            # buffer and set weight.data = pinned_buffer. So weight.data is
            # now on CPU in the pinned buffer. Re-alias state.CB to weight.data
            # instead of allocating a SECOND pinned buffer with the same data.
            if hasattr(module, 'state'):
                state = module.state
                
                # state.CB → alias to weight.data (already in pinned buffer)
                if state.CB is not None and state.CB.device.type != 'cpu':
                    if weight.data.device.type == 'cpu':
                        # weight.data already moved to pinned buffer — re-alias
                        state.CB = weight.data
                    else:
                        # Fallback: weight.data somehow not on CPU yet.
                        # This shouldn't happen in normal flow, but handle it.
                        key = f"__int8__{name}__state_CB"
                        if key in store:
                            store[key].copy_(state.CB, non_blocking=non_blocking)
                            state.CB = store[key]
                        else:
                            buf, _ = self._alloc_pinned_buffer(state.CB)
                            buf.copy_(state.CB, non_blocking=non_blocking)
                            store[key] = buf
                            state.CB = buf
                
                # state.SCB (scales — small tensor, no significant doubling)
                if state.SCB is not None and state.SCB.device.type != 'cpu':
                    key = f"__int8__{name}__state_SCB"
                    if key in store:
                        store[key].copy_(state.SCB, non_blocking=non_blocking)
                        state.SCB = store[key]
                    else:
                        buf, _ = self._alloc_pinned_buffer(state.SCB)
                        buf.copy_(state.SCB, non_blocking=non_blocking)
                        store[key] = buf
                        state.SCB = buf
    
    # ------------------------------------------------------------------
    # Initial placement
    # ------------------------------------------------------------------
    
    def setup_initial_placement(self) -> None:
        """
        Set up initial block placement based on config.
        
        For device_map models (BF16), we DETECT actual placement rather than moving blocks.
        For moveable models (NF4), we set up blocks based on config.
        
        Blocks 0 to (num_blocks - blocks_to_swap - 1) stay on GPU.
        Remaining blocks start on CPU/offload device.
        
        After placement, creates pre-allocated pinned CPU buffers for all
        swappable blocks to prevent CRT heap fragmentation during generation.
        """
        if not self.blocks:
            logger.warning("No blocks to set up")
            return
        
        blocks_to_swap = min(self.config.blocks_to_swap, self.num_blocks)
        
        # First, detect actual placement (important for device_map models)
        actual_placement, gpu_count, cpu_count = self._detect_actual_block_placement()
        
        if blocks_to_swap == 0:
            # Block swapping disabled - just record actual locations
            # For device_map models, blocks may be split across GPU/CPU/meta!
            # Meta blocks = CPU-offloaded by accelerate (served via hooks during forward)
            offloaded_count = cpu_count + sum(1 for d in actual_placement.values() if getattr(d, 'type', '') == 'meta')
            if offloaded_count > 0:
                logger.info(f"Block swap disabled, device_map placement: {gpu_count} blocks on GPU, {offloaded_count} CPU-offloaded (accelerate manages)")
                self.block_locations = actual_placement
                self.stats.blocks_currently_on_gpu = gpu_count
                self.stats.blocks_currently_on_cpu = offloaded_count
            else:
                logger.info("Block swap disabled - all blocks on GPU")
                for i, block in enumerate(self.blocks):
                    self.block_locations[i] = self.target_device
                self.stats.blocks_currently_on_gpu = self.num_blocks
            return
        
        # Calculate which blocks to offload
        # We offload the LAST N blocks (they run later in forward pass)
        swap_start = self.num_blocks - blocks_to_swap
        
        gpu_blocks = 0
        cpu_blocks = 0
        
        for i, block in enumerate(self.blocks):
            if i < swap_start:
                # Keep on GPU -- move there if not already
                current = actual_placement.get(i)
                if current != self.target_device:
                    # Block is on CPU/other device (e.g., INT8/BF16 loaded with device_map="cpu")
                    # Must actively move to GPU
                    self._move_block_to_device_raw(i, self.target_device, non_blocking=False)
                else:
                    self.block_locations[i] = self.target_device
                gpu_blocks += 1
            else:
                # Move to offload device (uses raw path since buffer store
                # doesn't exist yet)
                self._move_block_to_device_raw(i, self.offload_device)
                cpu_blocks += 1
        
        self.stats.blocks_currently_on_gpu = gpu_blocks
        self.stats.blocks_currently_on_cpu = cpu_blocks
        
        # Initialize CUDA stream for prefetching
        if self.config.prefetch_blocks > 0 and self.target_device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.target_device)
        
        # === Create pinned CPU buffer store ===
        # This MUST happen after blocks are moved to their initial devices
        # so we can snapshot the CPU tensors and swap param.data to pinned.
        self._create_cpu_buffer_store()
        
        # === Flush CUDA caching allocator ===
        # During initial placement, blocks were moved GPU<->CPU which causes
        # PyTorch's CUDA caching allocator to hoard freed GPU memory. Without
        # this flush, the allocator can hold 70-80GB of "reserved but unused"
        # VRAM, leaving no room for inference activations (OOM on first step).
        if self.target_device.type == "cuda":
            before_reserved = torch.cuda.memory_reserved(self.target_device) / 1024**3
            torch.cuda.empty_cache()
            after_reserved = torch.cuda.memory_reserved(self.target_device) / 1024**3
            freed = before_reserved - after_reserved
            if freed > 0.1:
                logger.info(f"  Flushed CUDA cache: reclaimed {freed:.1f}GB VRAM "
                           f"(reserved: {before_reserved:.1f}GB -> {after_reserved:.1f}GB)")
        
        # Calculate memory savings
        if cpu_blocks > 0:
            saved_gb = self.gb_per_block * cpu_blocks
            
            logger.info(f"Block swap setup complete:")
            logger.info(f"  Blocks on GPU: {gpu_blocks} (indices 0-{swap_start-1})")
            logger.info(f"  Blocks on {self.offload_device}: {cpu_blocks} (indices {swap_start}-{self.num_blocks-1})")
            logger.info(f"  Estimated VRAM saved: {saved_gb:.2f} GB")
    
    # ------------------------------------------------------------------
    # Block movement
    # ------------------------------------------------------------------
    
    def _move_nf4_block_to_device(
        self,
        block_idx: int,
        device: torch.device,
    ) -> float:
        """Move an NF4 block by iterating Params4bit individually.

        Params4bit.to(device, non_blocking=True) raises cudaErrorInvalidValue
        when moving CUDA→CPU.  We work around this by moving each parameter
        with non_blocking=False and explicitly handling quant_state tensors.

        This is slower than block.to() but is the only safe way for NF4.
        """
        block = self.blocks[block_idx]
        current_device = self.block_locations.get(block_idx)
        if current_device == device:
            return 0.0

        start_time = time.time()

        try:
            from bitsandbytes.nn import Params4bit
        except ImportError:
            Params4bit = None

        for param in block.parameters():
            if Params4bit is not None and isinstance(param, Params4bit):
                # Params4bit stores quantized data in .data and metadata in
                # .quant_state (QuantState object with absmax, code, offset).
                # We must move both; quant_state.to() handles its internals.
                if param.data.device != device:
                    param.data = param.data.to(device)
                if hasattr(param, 'quant_state') and param.quant_state is not None:
                    qs = param.quant_state
                    if hasattr(qs, 'to'):
                        qs.to(device)
                    else:
                        # QuantState may store tensors as plain attributes
                        for attr_name in ('absmax', 'code', 'offset', 'state2', 'blocksize', 'dtype'):
                            attr_val = getattr(qs, attr_name, None)
                            if attr_val is not None and hasattr(attr_val, 'to') and getattr(attr_val, 'device', None) != device:
                                setattr(qs, attr_name, attr_val.to(device))
            else:
                if param.device != device:
                    param.data = param.data.to(device)

        # Also move any plain buffers
        for buf in block.buffers():
            if buf.device != device:
                buf.data = buf.data.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.time() - start_time
        self.block_locations[block_idx] = device
        self.stats.total_swap_time_seconds += elapsed

        if device == self.target_device:
            self.stats.total_swaps_to_gpu += 1
            self.stats.blocks_currently_on_gpu += 1
            self.stats.blocks_currently_on_cpu -= 1
        else:
            self.stats.total_swaps_to_cpu += 1
            self.stats.blocks_currently_on_gpu -= 1
            self.stats.blocks_currently_on_cpu += 1

        return elapsed

    def _move_block_to_device_raw(
        self,
        block_idx: int,
        device: torch.device,
        non_blocking: bool = None,
    ) -> float:
        """Move a block using standard block.to() - no buffer store.

        Used during initial placement (before buffer store exists) and as
        fallback when a block has no buffer store entry.
        """
        if block_idx >= len(self.blocks):
            return 0.0

        block = self.blocks[block_idx]
        current_device = self.block_locations.get(block_idx)
        if current_device == device:
            return 0.0

        if non_blocking is None:
            non_blocking = self.config.use_non_blocking

        start_time = time.time()

        if self._has_nf4_layers:
            # NF4: block.to() with non_blocking=True crashes on CUDA→CPU.
            # Use per-parameter move instead.
            elapsed = self._move_nf4_block_to_device(block_idx, device)
            return elapsed

        block.to(device, non_blocking=non_blocking)
        self._fix_int8_state_devices(block, device)

        if not non_blocking and device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.time() - start_time
        self.block_locations[block_idx] = device
        self.stats.total_swap_time_seconds += elapsed

        if device == self.target_device:
            self.stats.total_swaps_to_gpu += 1
            self.stats.blocks_currently_on_gpu += 1
            self.stats.blocks_currently_on_cpu -= 1
        else:
            self.stats.total_swaps_to_cpu += 1
            self.stats.blocks_currently_on_gpu -= 1
            self.stats.blocks_currently_on_cpu += 1

        if self.config.debug:
            logger.debug(f"Block {block_idx} moved to {device} in {elapsed*1000:.1f}ms (raw)")

        return elapsed
    
    def _move_block_to_device(
        self,
        block_idx: int,
        device: torch.device,
        non_blocking: bool = None
    ) -> float:
        """
        Move a block to specified device, using pinned buffer store when available.
        
        GPU->CPU path: copies parameter data into pre-allocated pinned CPU buffers.
                       Zero new CPU allocations - prevents CRT heap fragmentation.
        CPU->GPU path: transfers from pinned buffers via DMA (2-3x faster than
                       pageable memory).
        Fallback:      standard block.to() if no buffer store for this block.
        
        Args:
            block_idx: Index of block to move
            device: Target device
            non_blocking: Use non-blocking transfer (None = use config)
            
        Returns:
            Time taken in seconds
        """
        if block_idx >= len(self.blocks):
            logger.warning(f"Block index {block_idx} out of range")
            return 0.0
        
        block = self.blocks[block_idx]
        current_device = self.block_locations.get(block_idx)
        
        if current_device == device:
            return 0.0  # Already on target device
        
        if non_blocking is None:
            non_blocking = self.config.use_non_blocking
        
        store = self._cpu_param_store.get(block_idx)
        
        if store is None:
            # No buffer store - fall back to raw block.to()
            return self._move_block_to_device_raw(block_idx, device, non_blocking)
        
        start_time = time.time()
        
        if device.type == 'cpu' or device == self.offload_device:
            # === BUFFERED GPU->CPU ===
            # Copy each tensor into its pre-allocated pinned buffer, then
            # point the param/buffer at the pinned tensor. The old GPU tensor
            # is released back to the CUDA caching allocator.
            for name, param in block.named_parameters():
                if name in store and param.data.device != device:
                    store[name].copy_(param.data, non_blocking=non_blocking)
                    param.data = store[name]
            
            for name, buf in block.named_buffers():
                key = f"__buffer__{name}"
                if key in store and buf.data.device != device:
                    store[key].copy_(buf.data, non_blocking=non_blocking)
                    buf.data = store[key]
            
            if self._has_int8_layers:
                self._move_int8_to_cpu_buffered(block, store, non_blocking)
        
        else:
            # === BUFFERED CPU->GPU ===
            # Transfer from pinned CPU buffers via DMA. Each param.data already
            # points to a pinned buffer (set during _create_cpu_buffer_store or
            # previous GPU->CPU move), so .to(cuda, non_blocking=True) uses
            # the fast cudaMemcpyAsync pinned path.
            for name, param in block.named_parameters():
                if param.data.device != device:
                    param.data = param.data.to(device, non_blocking=non_blocking)
            
            for name, buf in block.named_buffers():
                if buf.data.device != device:
                    buf.data = buf.data.to(device, non_blocking=non_blocking)
            
            if self._has_int8_layers:
                self._fix_int8_state_devices(block, device)
        
        # Sync if blocking transfer to GPU
        if not non_blocking and device.type == "cuda":
            torch.cuda.synchronize(device)
        
        elapsed = time.time() - start_time
        
        # Update tracking
        self.block_locations[block_idx] = device
        self.stats.total_swap_time_seconds += elapsed
        
        if device == self.target_device:
            self.stats.total_swaps_to_gpu += 1
            self.stats.blocks_currently_on_gpu += 1
            self.stats.blocks_currently_on_cpu -= 1
        else:
            self.stats.total_swaps_to_cpu += 1
            self.stats.blocks_currently_on_gpu -= 1
            self.stats.blocks_currently_on_cpu += 1
        
        if self.config.debug:
            logger.debug(f"Block {block_idx} moved to {device} in {elapsed*1000:.1f}ms (buffered)")
        
        return elapsed
    
    # ------------------------------------------------------------------
    # Prepare / release / prefetch
    # ------------------------------------------------------------------
    
    def prepare_block(self, block_idx: int) -> None:
        """
        Prepare a block for execution (ensure it's on GPU).
        
        Call this before running the block's forward pass.
        Also triggers prefetch of upcoming blocks.
        
        Args:
            block_idx: Index of block about to run
        """
        if not self.blocks or self.config.blocks_to_swap == 0:
            return
        
        # Check if this is a swappable block
        swap_start = self.num_blocks - self.config.blocks_to_swap
        if block_idx < swap_start:
            return  # This block is always on GPU, nothing to do
        
        if self.config.debug:
            logger.info(f"prepare_block({block_idx})")
        
        # ALWAYS check for and wait on prefetch events first
        # This must happen BEFORE checking location because prefetch updates location optimistically
        if block_idx in self._prefetch_events:
            event = self._prefetch_events[block_idx]
            if event is not None:
                if self.config.debug:
                    logger.info(f"  Waiting for prefetch event...")
                event.synchronize()
            del self._prefetch_events[block_idx]
        else:
            # Block wasn't prefetched, need to do sync transfer
            current_device = self.block_locations.get(block_idx)
            if current_device != self.target_device:
                if self.config.debug:
                    logger.info(f"  Moving {current_device} -> {self.target_device}")
                self._move_block_to_device(block_idx, self.target_device, non_blocking=False)
        
        # Verify block is actually on GPU (sanity check)
        block = self.blocks[block_idx]
        try:
            first_param = next(block.parameters())
            if first_param.device != self.target_device:
                logger.warning(f"Block {block_idx} not on {self.target_device}, forcing move!")
                block.to(self.target_device)
                torch.cuda.synchronize()
                self.block_locations[block_idx] = self.target_device
        except StopIteration:
            pass
        
        # Trigger prefetch of upcoming blocks
        self._prefetch_upcoming(block_idx)


    def release_block(self, block_idx: int) -> None:
        """
        Release a block after execution (move back to CPU if swapped).
        
        Call this after running the block's forward pass.
        
        Args:
            block_idx: Index of block that just finished
        """
        if not self.blocks or self.config.blocks_to_swap == 0:
            return
        
        # Only move back blocks that should be swapped
        swap_start = self.num_blocks - self.config.blocks_to_swap
        
        if block_idx >= swap_start:
            self._move_block_to_device(block_idx, self.offload_device)
    
    def _prefetch_upcoming(self, current_idx: int) -> None:
        """
        Prefetch upcoming blocks asynchronously using pinned buffers.
        
        When buffer store is available, uses param-level .to() from pinned
        memory which enables true async DMA via cudaMemcpyAsync.
        
        Args:
            current_idx: Currently executing block index
        """
        if self._prefetch_stream is None or self.config.prefetch_blocks == 0:
            return
        
        swap_start = self.num_blocks - self.config.blocks_to_swap
        
        for offset in range(1, self.config.prefetch_blocks + 1):
            prefetch_idx = current_idx + offset
            
            if prefetch_idx >= self.num_blocks:
                break
            
            # Only prefetch blocks that are swapped
            if prefetch_idx < swap_start:
                continue
            
            # Skip if already on GPU or already being prefetched
            if self.block_locations.get(prefetch_idx) == self.target_device:
                continue
            if prefetch_idx in self._prefetch_events:
                continue
            
            # Async prefetch
            block = self.blocks[prefetch_idx]
            store = self._cpu_param_store.get(prefetch_idx)

            if self._has_nf4_layers:
                # NF4: block.to(non_blocking=True) crashes (cudaErrorInvalidValue).
                # Cannot use async streams — fall back to synchronous per-param move.
                self._move_nf4_block_to_device(prefetch_idx, self.target_device)
                self._prefetch_events[prefetch_idx] = None  # no event needed
                self.block_locations[prefetch_idx] = self.target_device
                self.stats.total_swaps_to_gpu += 1
            else:
                with torch.cuda.stream(self._prefetch_stream):
                    if self._has_int8_layers:
                        # INT8: must use block.to() + aliasing fix.
                        # Pinned buffer store is skipped for INT8 (WDDM issue).
                        block.to(self.target_device, non_blocking=True)
                        self._fix_int8_state_devices(block, self.target_device)
                    elif store is not None:
                        # Param-level transfers from pinned buffers (fast DMA)
                        for name, param in block.named_parameters():
                            if param.data.device != self.target_device:
                                param.data = param.data.to(self.target_device, non_blocking=True)

                        for name, buf in block.named_buffers():
                            if buf.data.device != self.target_device:
                                buf.data = buf.data.to(self.target_device, non_blocking=True)

                        if self._has_int8_layers:
                            self._fix_int8_state_devices(block, self.target_device)
                    else:
                        # Fallback: standard block.to()
                        block.to(self.target_device, non_blocking=True)
                        self._fix_int8_state_devices(block, self.target_device)

                    # Record event for synchronization
                    event = torch.cuda.Event()
                    event.record(self._prefetch_stream)
                    self._prefetch_events[prefetch_idx] = event

                    self.block_locations[prefetch_idx] = self.target_device
                    self.stats.total_swaps_to_gpu += 1
            
            if self.config.debug:
                logger.debug(f"Prefetching block {prefetch_idx}")
    
    # ------------------------------------------------------------------
    # Hook installation / removal
    # ------------------------------------------------------------------
    
    def install_hooks(self) -> bool:
        """
        Install forward hooks on transformer blocks for automatic swapping.
        
        This enables automatic block swapping during model forward pass.
        Hooks call prepare_block() before and release_block() after each block.
        
        Returns:
            True if hooks were installed successfully
        """
        if self._hooks_installed:
            logger.warning("Hooks already installed")
            return True
        
        if not self.blocks:
            logger.warning("No blocks found, cannot install hooks")
            return False
        
        if self.config.blocks_to_swap == 0:
            logger.info("Block swap disabled (blocks_to_swap=0), skipping hook installation")
            return True
        
        self._hook_handles = []
        self._first_run_logged = False  # For one-time diagnostic logging
        
        for i, block in enumerate(self.blocks):
            # Pre-hook: prepare block before forward AND fix input tensor device.
            # Uses with_kwargs=True to also fix keyword arg tensors (attention_mask,
            # position_ids, custom_pos_emb) that must be on the same device as
            # hidden_states for the block's forward pass.
            def make_pre_hook(block_idx):
                def hook(module, args, kwargs):
                    self.prepare_block(block_idx)
                    # One-time diagnostic log on first block of first diffusion step
                    if not self._first_run_logged and block_idx == 0:
                        self._first_run_logged = True
                        input_dev = args[0].device if args and isinstance(args[0], torch.Tensor) else "N/A"
                        block_dev = next(module.parameters()).device
                        logger.info(f"  [BlockSwap] First run diagnostic: block_0 params on {block_dev}, "
                                   f"hidden_states on {input_dev}, target={self.target_device}")
                        # Deep INT8 state diagnostic
                        if self._has_int8_layers:
                            try:
                                from bitsandbytes.nn import Linear8bitLt
                                for name, mod in module.named_modules():
                                    if isinstance(mod, Linear8bitLt):
                                        w = mod.weight
                                        s = mod.state
                                        cb_dev = w.CB.device if (hasattr(w, 'CB') and w.CB is not None) else 'None'
                                        scb_dev = w.SCB.device if (hasattr(w, 'SCB') and w.SCB is not None) else 'None'
                                        s_cb_dev = s.CB.device if s.CB is not None else 'None'
                                        s_scb_dev = s.SCB.device if s.SCB is not None else 'None'
                                        logger.info(f"    [INT8] {name}: weight.data={w.data.device}, "
                                                   f"weight.CB={cb_dev}, weight.SCB={scb_dev}, "
                                                   f"state.CB={s_cb_dev}, state.SCB={s_scb_dev}, "
                                                   f"threshold={s.threshold}")
                                        break  # Just log first INT8 layer to keep output brief
                            except Exception as e:
                                logger.warning(f"    [INT8] diagnostic failed: {e}")
                    
                    # Ensure ALL input tensors are on the target GPU device.
                    # With INT8 models loaded via device_map="cpu", the pipeline
                    # may create input tensors on CPU. The block is now on GPU,
                    # so all inputs must match.
                    modified = False
                    args = list(args)
                    for i_arg, arg in enumerate(args):
                        if isinstance(arg, torch.Tensor) and arg.device != self.target_device:
                            args[i_arg] = arg.to(self.target_device)
                            modified = True
                    
                    for key, val in kwargs.items():
                        if isinstance(val, torch.Tensor) and val.device != self.target_device:
                            kwargs[key] = val.to(self.target_device)
                            modified = True
                        elif isinstance(val, tuple):
                            # Handle tuple kwargs like custom_pos_emb=(cos, sin)
                            new_val = tuple(
                                v.to(self.target_device) if isinstance(v, torch.Tensor) and v.device != self.target_device else v
                                for v in val
                            )
                            if any(isinstance(v, torch.Tensor) for v in val):
                                kwargs[key] = new_val
                                modified = True
                    
                    if modified and not hasattr(self, '_device_fix_logged'):
                        self._device_fix_logged = True
                        orig_dev = "cpu"  # We know it was moved
                        logger.info(f"  [BlockSwap] Moving input tensors from {orig_dev} "
                                   f"to {self.target_device} (block {block_idx})")
                    
                    return tuple(args), kwargs
                return hook
            
            # Post-hook: release block after forward
            def make_post_hook(block_idx):
                def hook(module, args, output):
                    self.release_block(block_idx)
                    return output
                return hook
            
            pre_handle = block.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True)
            post_handle = block.register_forward_hook(make_post_hook(i))
            
            self._hook_handles.extend([pre_handle, post_handle])
        
        self._hooks_installed = True
        logger.info(f"Installed block swap hooks on {len(self.blocks)} blocks")
        
        # Install safety-net hooks on INT8 Linear8bitLt layers.
        # These hooks run BEFORE each Linear8bitLt.forward() and fix any
        # weight.CB/SCB or state.CB/SCB device mismatches that block.to()
        # missed (bitsandbytes Module._apply bug).
        if self._has_int8_layers:
            self._install_int8_guard_hooks()
        
        return True
    
    def _install_int8_guard_hooks(self) -> None:
        """
        Install safety-net pre-forward hooks on every Linear8bitLt layer
        to fix device mismatches right before the INT8 forward pass.
        
        This catches any case where _fix_int8_state_devices was insufficient,
        such as edge cases in the bitsandbytes Int8Params lifecycle.
        
        The hook is lightweight: it only checks device attributes (no copies
        unless a mismatch is found). On the first mismatch found, it logs
        a warning to help with debugging.
        """
        try:
            from bitsandbytes.nn import Linear8bitLt
        except ImportError:
            return
        
        guard_count = 0
        _guard_logged = [False]  # mutable flag for closure
        
        def make_int8_guard_hook(layer_name):
            def hook(module, args):
                # Determine target device from the input tensor
                if args and isinstance(args[0], torch.Tensor):
                    target = args[0].device
                else:
                    target = self.target_device
                
                fixed = False
                weight = module.weight
                
                # Fix weight.CB/SCB (before init_8bit_state)
                if hasattr(weight, 'CB') and weight.CB is not None:
                    if weight.CB.device != target:
                        weight.CB = weight.CB.to(target)
                        fixed = True
                if hasattr(weight, 'SCB') and weight.SCB is not None:
                    if weight.SCB.device != target:
                        weight.SCB = weight.SCB.to(target)
                        fixed = True
                
                # Fix state.CB/SCB (after init_8bit_state)
                if hasattr(module, 'state'):
                    state = module.state
                    if state.CB is not None and state.CB.device != target:
                        state.CB = state.CB.to(target)
                        fixed = True
                    if state.SCB is not None and state.SCB.device != target:
                        state.SCB = state.SCB.to(target)
                        fixed = True
                
                if fixed and not _guard_logged[0]:
                    _guard_logged[0] = True
                    logger.warning(f"  [INT8 guard] Fixed device mismatch in {layer_name} "
                                 f"-> moved to {target} (bitsandbytes .to() workaround)")
            return hook
        
        for block_idx, block in enumerate(self.blocks):
            for name, module in block.named_modules():
                if isinstance(module, Linear8bitLt):
                    full_name = f"block_{block_idx}.{name}"
                    handle = module.register_forward_pre_hook(make_int8_guard_hook(full_name))
                    self._hook_handles.append(handle)
                    guard_count += 1
        
        logger.info(f"  Installed {guard_count} INT8 guard hooks across {len(self.blocks)} blocks")
    
    def remove_hooks(self) -> int:
        """
        Remove all installed block swap hooks.
        
        Returns:
            Number of hooks removed
        """
        removed = 0
        
        for handle in self._hook_handles:
            handle.remove()
            removed += 1
        
        self._hook_handles = []
        self._hooks_installed = False
        
        if removed > 0:
            logger.info(f"Removed {removed} block swap hooks")
        
        return removed
    
    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    
    def cleanup(self) -> None:
        """
        Full cleanup: remove hooks, release buffers, break circular references.
        
        MUST be called before deleting the model to prevent RAM leaks.
        Without this, the circular reference chain:
            model -> _block_swap_manager -> model (via self.model)
            model -> _block_swap_manager -> blocks[] -> model.model.layers[i]
        keeps all transformer block tensors alive in RAM even after
        `del model`, because Python's gc can't always collect cycles
        involving C-extension objects (torch tensors, bitsandbytes Int8Params).
        
        This method:
        1. Removes all forward hooks
        2. Releases pinned CPU buffer store (frees pinned memory)
        3. Clears the blocks list (drops refs to model.model.layers[i])
        4. Clears CUDA streams and prefetch events
        5. Nulls the model reference (breaks the circular ref)
        6. Clears block location tracking
        """
        # Step 1: Remove hooks first (they hold closures referencing self)
        if self._hooks_installed:
            self.remove_hooks()
        
        # Step 2: Release pinned CPU buffer store
        # Each buffer is a pinned tensor - clearing the dict releases the
        # pinned memory back to the CUDA host allocator.
        if self._cpu_param_store:
            num_buffers = sum(len(s) for s in self._cpu_param_store.values())
            total_bytes = sum(
                buf.numel() * buf.element_size()
                for store in self._cpu_param_store.values()
                for buf in store.values()
            )
            self._cpu_param_store.clear()
            logger.info(f"  Released pinned CPU buffer store: "
                        f"{num_buffers} buffers, {total_bytes / 1024**3:.2f} GB")
        
        # Step 3: Clear block references (these hold the actual layer tensors in RAM)
        num_blocks = len(self.blocks)
        self.blocks.clear() if isinstance(self.blocks, list) else None
        self.blocks = []
        
        # Step 4: Clear CUDA streams and prefetch events
        if self._prefetch_events:
            for event in self._prefetch_events.values():
                try:
                    event.synchronize()
                except Exception:
                    pass
            self._prefetch_events.clear()
        self._prefetch_stream = None
        
        # Step 5: Clear block location tracking
        self.block_locations.clear()
        
        # Step 6: Null the model reference (breaks the circular ref)
        self.model = None
        
        logger.info(f"BlockSwapManager cleanup complete: released refs to {num_blocks} blocks")
    
    @property
    def hooks_installed(self) -> bool:
        """Check if hooks are currently installed."""
        return self._hooks_installed
    
    # ------------------------------------------------------------------
    # Bulk move utilities
    # ------------------------------------------------------------------
    
    def move_all_to_gpu(self) -> float:
        """
        Move all blocks to GPU.
        
        Used for restoring after soft unload.
        
        Returns:
            Time taken in seconds
        """
        if not self.blocks:
            return 0.0
        
        total_time = 0.0
        for i in range(self.num_blocks):
            if self.block_locations.get(i) != self.target_device:
                total_time += self._move_block_to_device(i, self.target_device)
        
        torch.cuda.synchronize()
        logger.info(f"All {self.num_blocks} blocks moved to GPU in {total_time:.2f}s")
        return total_time
    
    def move_all_to_cpu(self) -> float:
        """
        Move all blocks to CPU.
        
        Used for soft unload.
        
        Returns:
            Time taken in seconds
        """
        if not self.blocks:
            return 0.0
        
        total_time = 0.0
        for i in range(self.num_blocks):
            if self.block_locations.get(i) != self.offload_device:
                total_time += self._move_block_to_device(i, self.offload_device)
        
        logger.info(f"All {self.num_blocks} blocks moved to {self.offload_device} in {total_time:.2f}s")
        return total_time
    
    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    
    def get_memory_summary(self) -> Dict:
        """Get memory usage summary."""
        if not self.blocks:
            return {"error": "No blocks found"}
        
        gpu_blocks = sum(1 for loc in self.block_locations.values() if loc == self.target_device)
        cpu_blocks = self.num_blocks - gpu_blocks
        
        pinned_bytes = sum(
            buf.numel() * buf.element_size()
            for store in self._cpu_param_store.values()
            for buf in store.values()
        ) if self._cpu_param_store else 0
        
        return {
            "total_blocks": self.num_blocks,
            "blocks_on_gpu": gpu_blocks,
            "blocks_on_cpu": cpu_blocks,
            "bytes_per_block": self.bytes_per_block,
            "gb_per_block": self.gb_per_block,
            "gpu_memory_gb": gpu_blocks * self.gb_per_block,
            "cpu_memory_gb": cpu_blocks * self.gb_per_block,
            "pinned_buffer_gb": pinned_bytes / 1024**3,
            "hooks_installed": self._hooks_installed,
            "stats": str(self.stats),
        }
    
    def reset_stats(self) -> None:
        """Reset swap statistics."""
        self.stats = BlockSwapStats(
            blocks_currently_on_gpu=sum(
                1 for loc in self.block_locations.values() 
                if loc == self.target_device
            ),
            blocks_currently_on_cpu=sum(
                1 for loc in self.block_locations.values() 
                if loc != self.target_device
            )
        )
    
    @contextmanager
    def swap_context(self):
        """
        Context manager for block swap during forward pass.
        
        Usage:
            with manager.swap_context():
                output = model(input)
        """
        # Reset stats for this forward pass
        self.reset_stats()
        
        try:
            yield self
        finally:
            # Clean up any pending prefetch events
            for event in self._prefetch_events.values():
                event.synchronize()
            self._prefetch_events.clear()
    
    def __repr__(self) -> str:
        return (
            f"BlockSwapManager(blocks={self.num_blocks}, "
            f"swap={self.config.blocks_to_swap}, "
            f"gpu={self.stats.blocks_currently_on_gpu}, "
            f"hooks={'on' if self._hooks_installed else 'off'})"
        )


def calculate_blocks_to_swap(
    available_vram_gb: float,
    model_vram_gb: float,
    inference_vram_gb: float,
    num_blocks: int,
    gb_per_block: float,
    reserve_vram_gb: float = 2.0,
    safety_margin: float = 0.9
) -> Tuple[int, Dict[str, float]]:
    """
    Calculate optimal number of blocks to swap based on VRAM constraints.
    
    Args:
        available_vram_gb: Total available VRAM
        model_vram_gb: VRAM used by model (non-block parts)
        inference_vram_gb: VRAM needed for inference activations
        num_blocks: Total number of transformer blocks
        gb_per_block: VRAM per block
        reserve_vram_gb: VRAM to reserve for downstream nodes
        safety_margin: Safety factor (0.9 = use 90% of calculated capacity)
        
    Returns:
        Tuple of (blocks_to_swap, details_dict)
    """
    # Calculate total needed
    block_vram = num_blocks * gb_per_block
    total_needed = model_vram_gb + inference_vram_gb + reserve_vram_gb
    
    # Apply safety margin to available VRAM
    usable_vram = available_vram_gb * safety_margin
    
    # Calculate deficit
    deficit = total_needed - usable_vram
    
    details = {
        "available_vram_gb": available_vram_gb,
        "usable_vram_gb": usable_vram,
        "model_vram_gb": model_vram_gb,
        "inference_vram_gb": inference_vram_gb,
        "reserve_vram_gb": reserve_vram_gb,
        "total_needed_gb": total_needed,
        "block_vram_gb": block_vram,
        "gb_per_block": gb_per_block,
        "deficit_gb": max(0, deficit),
    }
    
    if deficit <= 0:
        # No swapping needed
        details["blocks_to_swap"] = 0
        details["reason"] = "Sufficient VRAM"
        return 0, details
    
    # Calculate how many blocks to swap
    blocks_to_swap = int((deficit / gb_per_block) + 0.999)  # Round up
    blocks_to_swap = min(blocks_to_swap, num_blocks - 1)  # Keep at least 1 on GPU
    blocks_to_swap = max(blocks_to_swap, 0)
    
    details["blocks_to_swap"] = blocks_to_swap
    details["vram_freed_gb"] = blocks_to_swap * gb_per_block
    details["reason"] = f"Deficit of {deficit:.1f}GB requires swapping {blocks_to_swap} blocks"
    
    return blocks_to_swap, details


# Legacy function for backwards compatibility
def install_block_swap_hooks(model: Any, manager: BlockSwapManager) -> None:
    """
    Install forward hooks on transformer blocks for automatic swapping.
    
    DEPRECATED: Use manager.install_hooks() instead.
    
    Args:
        model: The model containing transformer blocks
        manager: BlockSwapManager instance
    """
    logger.warning("install_block_swap_hooks() is deprecated. Use manager.install_hooks() instead.")
    manager.install_hooks()


def remove_block_swap_hooks(model: Any) -> int:
    """
    Remove block swap hooks from model.
    
    DEPRECATED: Use manager.remove_hooks() instead.
    
    Args:
        model: The model to remove hooks from
        
    Returns:
        Number of hooks removed
    """
    logger.warning("remove_block_swap_hooks() is deprecated. Use manager.remove_hooks() instead.")
    removed = 0
    
    for module in model.modules():
        # Clear forward hooks
        if hasattr(module, '_forward_hooks'):
            hooks_to_remove = list(module._forward_hooks.keys())
            for handle_id in hooks_to_remove:
                del module._forward_hooks[handle_id]
                removed += 1
        
        # Clear pre-hooks
        if hasattr(module, '_forward_pre_hooks'):
            hooks_to_remove = list(module._forward_pre_hooks.keys())
            for handle_id in hooks_to_remove:
                del module._forward_pre_hooks[handle_id]
                removed += 1
    
    if removed > 0:
        logger.info(f"Removed {removed} block swap hooks")
    
    return removed

"""
HunyuanImage-3.0 Block Swap Manager - V2 Unified Node Support

Kijai-style explicit block swapping for transformer blocks.
Moves blocks between GPU and CPU during forward pass to enable
large generations on limited VRAM.

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
        
        logger.info(f"BlockSwapManager initialized: {self.num_blocks} blocks found")
        logger.info(f"  Config: swap={config.blocks_to_swap}, prefetch={config.prefetch_blocks}")
        if self._has_int8_layers:
            logger.info(f"  INT8 quantized layers detected - will handle state.CB/SCB during swaps")
    
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
    
    def _fix_int8_state_devices(self, block: nn.Module, device: torch.device) -> None:
        """
        Fix bitsandbytes Linear8bitLt weight and state device after block.to().
        
        bitsandbytes has a bug where block.to(device) does NOT move weight.CB,
        weight.SCB, state.CB, or state.SCB to the target device. This is because:
        
        1. Module._apply(fn) calls fn(param) → Int8Params.to() which creates a NEW
           Int8Params with CB/SCB on the target device, but Module._apply only copies
           param.data (the raw tensor), discarding the new CB/SCB attributes.
        2. Linear8bitLt.to() overrides .to() to handle state.CB/SCB, but this override
           is never called when a PARENT module calls .to() (Module._apply recurses
           via child._apply, not child.to()).
        
        The INT8 lifecycle:
        - Before first forward: weight.CB has the compressed buffer, state.CB is None
        - After first forward: init_8bit_state() moves weight.CB → state.CB, weight.CB = None
        - After matmul: weight.data is set to state.CB (they share the same tensor)
        
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
                weight = module.weight
                if hasattr(weight, 'CB') and weight.CB is not None and weight.CB.device != device:
                    if self.config.debug or not hasattr(self, '_int8_fix_logged'):
                        logger.info(f"  [INT8 fix] {name}: weight.CB {weight.CB.device} → {device}")
                    weight.CB = weight.CB.to(device)
                    fixed_count += 1
                if hasattr(weight, 'SCB') and weight.SCB is not None and weight.SCB.device != device:
                    if self.config.debug or not hasattr(self, '_int8_fix_logged'):
                        logger.info(f"  [INT8 fix] {name}: weight.SCB {weight.SCB.device} → {device}")
                    weight.SCB = weight.SCB.to(device)
                    fixed_count += 1
                
                # Fix state.CB/SCB (after first forward, these hold the compressed data)
                if hasattr(module, 'state'):
                    state = module.state
                    if state.CB is not None and state.CB.device != device:
                        if self.config.debug or not hasattr(self, '_int8_fix_logged'):
                            logger.info(f"  [INT8 fix] {name}: state.CB {state.CB.device} → {device}")
                        state.CB = state.CB.to(device)
                        fixed_count += 1
                    if state.SCB is not None and state.SCB.device != device:
                        if self.config.debug or not hasattr(self, '_int8_fix_logged'):
                            logger.info(f"  [INT8 fix] {name}: state.SCB {state.SCB.device} → {device}")
                        state.SCB = state.SCB.to(device)
                        fixed_count += 1
        
        if fixed_count > 0 and not hasattr(self, '_int8_fix_logged'):
            self._int8_fix_logged = True
            logger.info(f"  [INT8 fix] Fixed {fixed_count} INT8 tensor(s) for this block → {device}")
    
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
    
    def setup_initial_placement(self) -> None:
        """
        Set up initial block placement based on config.
        
        For device_map models (BF16), we DETECT actual placement rather than moving blocks.
        For moveable models (NF4), we set up blocks based on config.
        
        Blocks 0 to (num_blocks - blocks_to_swap - 1) stay on GPU.
        Remaining blocks start on CPU/offload device.
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
                # Keep on GPU — move there if not already
                current = actual_placement.get(i)
                if current != self.target_device:
                    # Block is on CPU/other device (e.g., INT8/BF16 loaded with device_map="cpu")
                    # Must actively move to GPU
                    self._move_block_to_device(i, self.target_device, non_blocking=False)
                else:
                    self.block_locations[i] = self.target_device
                gpu_blocks += 1
            else:
                # Move to offload device
                self._move_block_to_device(i, self.offload_device)
                cpu_blocks += 1
        
        self.stats.blocks_currently_on_gpu = gpu_blocks
        self.stats.blocks_currently_on_cpu = cpu_blocks
        
        # Initialize CUDA stream for prefetching
        if self.config.prefetch_blocks > 0 and self.target_device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.target_device)
        
        # Calculate memory savings
        if cpu_blocks > 0:
            saved_gb = self.gb_per_block * cpu_blocks
            
            logger.info(f"Block swap setup complete:")
            logger.info(f"  Blocks on GPU: {gpu_blocks} (indices 0-{swap_start-1})")
            logger.info(f"  Blocks on {self.offload_device}: {cpu_blocks} (indices {swap_start}-{self.num_blocks-1})")
            logger.info(f"  Estimated VRAM saved: {saved_gb:.2f} GB")
    
    def _move_block_to_device(
        self,
        block_idx: int,
        device: torch.device,
        non_blocking: bool = None
    ) -> float:
        """
        Move a block to specified device.
        
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
        
        start_time = time.time()
        
        # Move all parameters and buffers
        block.to(device, non_blocking=non_blocking)
        
        # Fix INT8 bitsandbytes state tensors that block.to() missed
        # (Module._apply doesn't call Linear8bitLt.to() on children)
        self._fix_int8_state_devices(block, device)
        
        # Sync if blocking transfer
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
            logger.debug(f"Block {block_idx} moved to {device} in {elapsed*1000:.1f}ms")
        
        return elapsed
    
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
            if self.config.debug:
                logger.info(f"  Waiting for prefetch event...")
            self._prefetch_events[block_idx].synchronize()
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
        Prefetch upcoming blocks asynchronously.
        
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
            with torch.cuda.stream(self._prefetch_stream):
                block = self.blocks[prefetch_idx]
                block.to(self.target_device, non_blocking=True)
                
                # Fix INT8 state tensors that block.to() missed
                self._fix_int8_state_devices(block, self.target_device)
                
                # Record event for synchronization
                event = torch.cuda.Event()
                event.record(self._prefetch_stream)
                self._prefetch_events[prefetch_idx] = event
                
                self.block_locations[prefetch_idx] = self.target_device
                self.stats.total_swaps_to_gpu += 1
            
            if self.config.debug:
                logger.debug(f"Prefetching block {prefetch_idx}")
    
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
                                 f"→ moved to {target} (bitsandbytes .to() workaround)")
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
    
    @property
    def hooks_installed(self) -> bool:
        """Check if hooks are currently installed."""
        return self._hooks_installed
    
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
    
    def get_memory_summary(self) -> Dict:
        """Get memory usage summary."""
        if not self.blocks:
            return {"error": "No blocks found"}
        
        gpu_blocks = sum(1 for loc in self.block_locations.values() if loc == self.target_device)
        cpu_blocks = self.num_blocks - gpu_blocks
        
        return {
            "total_blocks": self.num_blocks,
            "blocks_on_gpu": gpu_blocks,
            "blocks_on_cpu": cpu_blocks,
            "bytes_per_block": self.bytes_per_block,
            "gb_per_block": self.gb_per_block,
            "gpu_memory_gb": gpu_blocks * self.gb_per_block,
            "cpu_memory_gb": cpu_blocks * self.gb_per_block,
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

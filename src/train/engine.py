"""
Training engine for neural network option pricing.

Handles training loops, validation, checkpointing, AMP, gradient clipping,
and curriculum learning for composite loss functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Literal, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from ..exp.config import OptimCfg, LossCfg
from .losses import composite_loss, greeks_loss as greeks_loss_fn
from .collocation import sample_collocation_points, sample_boundary_points, sample_adaptive_collocation


class TrainingEngine:
    """
    Engine for training neural networks with composite loss and curriculum.
    
    Handles:
    - Training and validation loops
    - Automatic mixed precision (AMP)
    - Gradient clipping
    - Learning rate scheduling
    - Curriculum learning (staged loss introduction)
    - Checkpointing and early stopping
    
    Attributes:
        model: Neural network model.
        optimizer: PyTorch optimizer.
        scheduler: Learning rate scheduler (optional).
        scaler: GradScaler for AMP (optional).
        device: Device ('cpu' or 'cuda').
        logger: Logger instance.
        
    Example:
        >>> from ..models.nets import MLP
        >>> model = MLP(L=3, W=32)
        >>> optim_cfg = OptimCfg(lr=1e-3, epochs=100)
        >>> loss_cfg = LossCfg()
        >>> engine = TrainingEngine(model, optim_cfg, loss_cfg)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optim_cfg: OptimCfg,
        loss_cfg: LossCfg,
        device: str = "cpu",
        logger: Optional[logging.Logger] = None,
        model_type: Literal["BS", "LocalVol", "Heston"] = "BS",
        local_vol_fn: Optional[Callable] = None,
        heston_params: Optional[dict] = None,
    ):
        """
        Initialize training engine.
        
        Args:
            model: Neural network.
            optim_cfg: Optimizer configuration.
            loss_cfg: Loss configuration.
            device: Device.
            logger: Optional logger.
            model_type: Pricing model type for PDE operator dispatch.
            local_vol_fn: Torch-compatible sigma(K, T) callable for LocalVol PDE.
            heston_params: Dict of Heston parameters for Heston PDE.
        """
        self.model = model.to(device)
        self.optim_cfg = optim_cfg
        self.loss_cfg = loss_cfg
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        self.model_type = model_type
        self.local_vol_fn = local_vol_fn
        self.heston_params = heston_params
        
        # Create optimizer
        self.optimizer = self._create_optimizer(optim_cfg)
        
        # Create scheduler
        self.scheduler = self._create_scheduler(optim_cfg)
        
        # AMP scaler
        self.scaler = GradScaler() if optim_cfg.amp and device == "cuda" else None
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        # Epoch of the restored (best) checkpoint and first epoch of the final
        # curriculum stage; set by train().
        self.best_epoch: int | None = None
        self.final_stage_start: int = 0

        # Metrics history
        self.history = []
    
    def _create_optimizer(self, cfg: OptimCfg) -> optim.Optimizer:
        """Create optimizer from config."""
        if cfg.name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.name == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.name}")
    
    def _create_scheduler(self, cfg: OptimCfg) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if cfg.schedule == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs
            )
        elif cfg.schedule == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.epochs // 3, gamma=0.1
            )
        else:
            return None
    
    @staticmethod
    def _compute_greek_weights(loader) -> dict[str, float]:
        """Compute per-Greek inverse-median-absolute-value weights (Ch8)."""
        accum: dict[str, list] = {}
        for batch in loader:
            for key in ("delta", "gamma", "theta", "vega"):
                if key in batch:
                    accum.setdefault(key, []).append(batch[key].abs())
        weights: dict[str, float] = {}
        for key, tensors in accum.items():
            med = torch.cat(tensors).median().item()
            weights[key] = 1.0 / max(med, 1e-8)
        if weights:
            max_w = max(weights.values())
            weights = {k: v / max_w for k, v in weights.items()}
        return weights

    def _stage_duration(self) -> int:
        """Number of epochs per curriculum stage (equal split of the epoch budget)."""
        return max(1, self.optim_cfg.epochs // len(self.loss_cfg.curriculum))

    def _final_stage_start(self) -> int:
        """First epoch of the final curriculum stage (0 for a single-stage curriculum).

        Early stopping and best-checkpoint selection are restricted to epochs
        >= this value, so the restored model has been trained with every loss
        term of its configuration (price, Greeks and/or PDE residual).
        """
        return (len(self.loss_cfg.curriculum) - 1) * self._stage_duration()

    def _get_curriculum_weights(self, epoch: int) -> dict[str, float]:
        """
        Get loss weights for current curriculum stage.
        
        Args:
            epoch: Current epoch.
            
        Returns:
            Dict of loss weights.
        """
        stages = self.loss_cfg.curriculum

        # Determine current stage
        stage_duration = self._stage_duration()
        stage_idx = min(epoch // stage_duration, len(stages) - 1)
        stage = stages[stage_idx]
        
        # Parse stage string (e.g., "price+greeks+pde")
        weights = {
            'lambda_price': self.loss_cfg.lambda_price if 'price' in stage else 0.0,
            'lambda_greeks': self.loss_cfg.lambda_greeks if 'greeks' in stage else 0.0,
            'lambda_pde': self.loss_cfg.lambda_pde if 'pde' in stage else 0.0,
            'lambda_bc': self.loss_cfg.lambda_bc,
            'lambda_noarb': self.loss_cfg.lambda_noarb if 'noarb' in stage else 0.0,
        }
        
        return weights
    
    def train_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        K_range: tuple[float, float],
        T_range: tuple[float, float],
        r: float,
        sigma: float,
        S0: float = 1.0,
    ) -> dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            data_loader: DataLoader with (K, T, price, greeks).
            K_range: (K_min, K_max) for collocation.
            T_range: (T_min, T_max) for collocation.
            r: Risk-free rate.
            sigma: Volatility.
            S0: Initial underlying price.
            
        Returns:
            Dict of average losses.
        """
        self.model.train()
        epoch_losses = []
        
        # Get curriculum weights
        weights = self._get_curriculum_weights(self.epoch)
        
        for batch in data_loader:
            K_data = batch['K'].to(self.device)
            T_data = batch['T'].to(self.device)
            price_target = batch['price'].to(self.device)
            
            greeks_target = None
            if 'delta' in batch and weights['lambda_greeks'] > 0:
                greeks_target = {
                    'delta': batch['delta'].to(self.device),
                    'gamma': batch['gamma'].to(self.device),
                }
                if 'theta' in batch:
                    greeks_target['theta'] = batch['theta'].to(self.device)
                if 'vega' in batch:
                    greeks_target['vega'] = batch['vega'].to(self.device)
            
            n_colloc = self.optim_cfg.batch_collocation
            use_adaptive = (
                self.epoch >= 200
                and weights.get('lambda_pde', 0) > 0
            )
            if use_adaptive:
                n_sobol = n_colloc // 2
                n_adaptive = n_colloc - n_sobol
                K_sobol, T_sobol = sample_collocation_points(
                    n_sobol,
                    K_range[0], K_range[1],
                    T_range[0], T_range[1],
                    device=self.device,
                )
                K_adapt, T_adapt = sample_adaptive_collocation(
                    self.model,
                    n_adaptive * 4,
                    K_range[0], K_range[1],
                    T_range[0], T_range[1],
                    r, sigma,
                    device=self.device,
                    top_k=n_adaptive,
                    model_type=self.model_type,
                    local_vol_fn=self.local_vol_fn,
                    heston_params=self.heston_params,
                )
                K_colloc = torch.cat([K_sobol, K_adapt], dim=0)
                T_colloc = torch.cat([T_sobol, T_adapt], dim=0)
            else:
                K_colloc, T_colloc = sample_collocation_points(
                    n_colloc,
                    K_range[0], K_range[1],
                    T_range[0], T_range[1],
                    device=self.device,
                )
            
            # Sample boundary points
            K_bc, T_bc = sample_boundary_points(
                self.optim_cfg.batch_data // 4,
                K_range[0], K_range[1],
                T_boundary=T_range[0],
                device=self.device,
            )
            
            self.optimizer.zero_grad()
            
            gw = getattr(self, 'greek_weights', None)

            needs_higher_order_grads = (
                weights.get('lambda_pde', 0) > 0
                or weights.get('lambda_greeks', 0) > 0
                or weights.get('lambda_noarb', 0) > 0
            )
            use_amp = self.scaler is not None and not needs_higher_order_grads

            if use_amp:
                with autocast():
                    loss, loss_dict = composite_loss(
                        self.model, K_data, T_data, price_target, greeks_target,
                        K_colloc, T_colloc, K_bc, T_bc,
                        weights['lambda_price'], weights['lambda_greeks'],
                        weights['lambda_pde'], weights['lambda_bc'],
                        weights['lambda_noarb'], r, sigma, S0,
                        model_type=self.model_type,
                        local_vol_fn=self.local_vol_fn,
                        heston_params=self.heston_params,
                        greek_weights=gw,
                    )

                self.scaler.scale(loss).backward()

                if self.optim_cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.optim_cfg.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, loss_dict = composite_loss(
                    self.model, K_data, T_data, price_target, greeks_target,
                    K_colloc, T_colloc, K_bc, T_bc,
                    weights['lambda_price'], weights['lambda_greeks'],
                    weights['lambda_pde'], weights['lambda_bc'],
                    weights['lambda_noarb'], r, sigma, S0,
                    model_type=self.model_type,
                    local_vol_fn=self.local_vol_fn,
                    heston_params=self.heston_params,
                    greek_weights=gw,
                )
                
                loss.backward()
                
                if self.optim_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.optim_cfg.grad_clip
                    )
                
                self.optimizer.step()
            
            epoch_losses.append(loss_dict)
        
        # Average losses
        avg_losses = {
            key: sum(d.get(key, 0.0) for d in epoch_losses) / len(epoch_losses)
            for key in epoch_losses[0].keys()
        }
        
        return avg_losses
    
    def validate(
        self,
        data_loader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        """Validate model on validation set (price + Greeks MSE)."""
        self.model.eval()
        total_price_loss = 0.0
        total_greeks_loss = 0.0
        count = 0

        for batch in data_loader:
            K = batch['K'].to(self.device)
            T = batch['T'].to(self.device)
            price_target = batch['price'].to(self.device)

            with torch.no_grad():
                price_pred = self.model(K, T)
                loss = ((price_pred - price_target) ** 2).mean()
                total_price_loss += loss.item()

            if 'delta' in batch and hasattr(self.model, 'compute_greeks_for_loss'):
                greeks_target = {
                    'delta': batch['delta'].to(self.device),
                    'gamma': batch['gamma'].to(self.device),
                }
                if 'theta' in batch:
                    greeks_target['theta'] = batch['theta'].to(self.device)
                with torch.enable_grad():
                    pred_g = self.model.compute_greeks_for_loss(K, T)
                g_loss = greeks_loss_fn(pred_g, greeks_target, weights=getattr(self, 'greek_weights', None))
                total_greeks_loss += g_loss.item()

            count += 1

        avg_price = total_price_loss / max(count, 1)
        avg_greeks = total_greeks_loss / max(count, 1)
        
        return {
            'val_loss': avg_price + avg_greeks,
            'val_price_loss': avg_price,
            'val_greeks_loss': avg_greeks,
        }
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        K_range: tuple[float, float],
        T_range: tuple[float, float],
        r: float,
        sigma: float,
        S0: float = 1.0,
    ) -> pd.DataFrame:
        """
        Full training loop with validation and early stopping.

        The loss curriculum switches on its stages at equally spaced epochs
        (see ``_get_curriculum_weights``). Best-checkpoint selection and early
        stopping are restricted to the final curriculum stage: the patience
        counter starts at the first epoch of that stage, and the restored model
        is the checkpoint with the lowest validation loss within it. A model
        reported as Greek- or PDE-trained has therefore always been trained
        with those terms switched on.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            K_range, T_range: Collocation ranges.
            r, sigma: Model parameters.
            S0: Initial price.

        Returns:
            DataFrame with training history.
        """
        self.greek_weights = self._compute_greek_weights(train_loader)

        import copy
        self.final_stage_start = self._final_stage_start()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = None
        best_state_dict = None

        for epoch in range(self.optim_cfg.epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch(
                train_loader, K_range, T_range, r, sigma, S0
            )
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Combine metrics
            metrics = {'epoch': epoch, **train_metrics, **val_metrics}
            self.history.append(metrics)
            
            # Log
            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{self.optim_cfg.epochs}: "
                    f"train_loss={train_metrics.get('total', 0):.6f}, "
                    f"val_loss={val_metrics['val_loss']:.6f}"
                )
            
            # Learning rate schedule
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Best-model checkpointing and early stopping, final curriculum
            # stage only (earlier stages lack some of the configured loss terms)
            if epoch < self.final_stage_start:
                continue
            val_loss = val_metrics['val_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                self.patience_counter += 1

            if (self.optim_cfg.early_stop_patience is not None
                    and self.patience_counter >= self.optim_cfg.early_stop_patience):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            self.logger.info(
                f"Restored best final-stage model (epoch={self.best_epoch}, "
                f"final stage starts at epoch {self.final_stage_start}, "
                f"val_loss={self.best_val_loss:.6f})"
            )

        return pd.DataFrame(self.history)
    
    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }, path)
        
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        self.logger.info(f"Checkpoint loaded from {path}")


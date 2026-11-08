"""
Command-line interface for running option pricing experiments.

Provides three main commands:
- run: Execute a single experiment from a config file
- grid: Execute multiple experiments (grid search)
- report: Generate report for a completed run
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

from ..exp.config import ExperimentCfg
from ..exp.io_utils import load_config
from ..exp.runner import ExperimentRunner


def cmd_run(args: argparse.Namespace) -> int:
    """
    Run a single experiment.
    
    Args:
        args: Parsed arguments with 'config' field.
        
    Returns:
        Exit code (0 = success, 1 = failure).
    """
    try:
        print(f"Loading config: {args.config}")
        cfg = load_config(args.config)
        
        print(f"Starting experiment: {cfg.experiment_id}")
        runner = ExperimentRunner(cfg, device=args.device)
        
        runner.prepare()
        runner.train_and_eval()
        metrics = runner.report()
        
        print("\n=== Experiment Complete ===")
        print(f"Run directory: {runner.run_dir}")
        print(f"Final metrics:")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.6f}")
            else:
                print(f"  {key}: {val}")
        
        return 0
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_grid(args: argparse.Namespace) -> int:
    """
    Run multiple experiments from a directory.
    
    Args:
        args: Parsed arguments with 'dir' and 'match' fields.
        
    Returns:
        Exit code.
    """
    try:
        # Find matching config files
        config_dir = Path(args.dir)
        pattern = args.match or "*.json"
        config_files = list(config_dir.glob(pattern))
        
        if not config_files:
            print(f"No config files found matching: {config_dir / pattern}")
            return 1
        
        print(f"Found {len(config_files)} config files")
        
        # Run each experiment
        results = []
        
        for i, config_path in enumerate(config_files, 1):
            print(f"\n{'='*60}")
            print(f"Experiment {i}/{len(config_files)}: {config_path.name}")
            print(f"{'='*60}")
            
            try:
                cfg = load_config(config_path)
                runner = ExperimentRunner(cfg, device=args.device)
                
                runner.prepare()
                runner.train_and_eval()
                metrics = runner.report()
                
                results.append({
                    'experiment_id': cfg.experiment_id,
                    'config_file': str(config_path),
                    'run_dir': str(runner.run_dir),
                    'status': 'success',
                    **metrics,
                })
            
            except Exception as e:
                print(f"ERROR in {config_path.name}: {e}")
                results.append({
                    'experiment_id': config_path.stem,
                    'config_file': str(config_path),
                    'status': 'failed',
                    'error': str(e),
                })
        
        # Save grid summary
        summary_df = pd.DataFrame(results)
        summary_path = Path("runs") / "grid_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"Grid search complete: {len(results)} experiments")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*60}")
        
        # Print summary table
        print("\nResults:")
        success_count = sum(1 for r in results if r.get('status') == 'success')
        print(f"  Success: {success_count}/{len(results)}")
        
        if 'rmse_price' in summary_df.columns:
            print(f"\nPrice RMSE:")
            for _, row in summary_df.iterrows():
                if row['status'] == 'success':
                    print(f"  {row['experiment_id']}: {row.get('rmse_price', 'N/A'):.6f}")
        
        return 0
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """
    Generate report for a completed run.
    
    Args:
        args: Parsed arguments with 'run' field.
        
    Returns:
        Exit code.
    """
    try:
        run_dir = Path(args.run)
        
        if not run_dir.exists():
            print(f"Run directory not found: {run_dir}")
            return 1
        
        # Load config and summary
        cfg_path = run_dir / "cfg.json"
        summary_path = run_dir / "summary.json"
        
        if not cfg_path.exists():
            print(f"Config not found: {cfg_path}")
            return 1
        
        cfg = load_config(cfg_path)
        
        # Load summary if exists
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        else:
            summary = {}
        
        # Generate report using template (if available)
        print(f"=== Experiment Report ===")
        print(f"ID: {cfg.experiment_id}")
        print(f"Model: {cfg.model.name}")
        print(f"Seed: {cfg.seed}")
        print(f"\nArchitecture:")
        print(f"  Layers: {cfg.net.L}")
        print(f"  Width: {cfg.net.W}")
        print(f"  Activation: {cfg.net.activation}")
        print(f"\nTraining:")
        print(f"  Epochs: {cfg.optim.epochs}")
        print(f"  Learning rate: {cfg.optim.lr}")
        print(f"  Optimizer: {cfg.optim.name}")
        
        if 'metrics' in summary:
            print(f"\nMetrics:")
            for key, val in summary['metrics'].items():
                if isinstance(val, float):
                    print(f"  {key}: {val:.6f}")
                elif isinstance(val, dict):
                    print(f"  {key}:")
                    for k2, v2 in val.items():
                        if isinstance(v2, float):
                            print(f"    {k2}: {v2:.6f}")
                else:
                    print(f"  {key}: {val}")
        
        # List artifacts
        print(f"\nArtifacts:")
        for item in run_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(run_dir)
                print(f"  {rel_path}")
        
        return 0
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Option Pricing Experiment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python -m src.cli.main run --config configs/bs_softplus_full.json
  
  # Run activation ablation
  python -m src.cli.main grid --dir configs --match "act_ablation*.json"
  
  # Generate report
  python -m src.cli.main report --run runs/20251010_120000__bs_full
        """
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cpu/cuda, default: auto)',
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run single experiment')
    run_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config JSON file',
    )
    
    # Grid command
    grid_parser = subparsers.add_parser('grid', help='Run multiple experiments')
    grid_parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Directory containing config files',
    )
    grid_parser.add_argument(
        '--match',
        type=str,
        default='*.json',
        help='Glob pattern to match config files (default: *.json)',
    )
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate report for run')
    report_parser.add_argument(
        '--run',
        type=str,
        required=True,
        help='Path to run directory',
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Dispatch to command handler
    if args.command == 'run':
        return cmd_run(args)
    elif args.command == 'grid':
        return cmd_grid(args)
    elif args.command == 'report':
        return cmd_report(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())


import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    print(' '.join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main():
    parser = argparse.ArgumentParser(description='Kaggle training helper for DINOv3 + Mask2Former')
    parser.add_argument('--dataset-root', required=True, help='Foot Ulcer Segmentation Challenge root')
    parser.add_argument('--coco-out', required=True, help='Output directory for COCO annotations')
    parser.add_argument('--output-dir', required=True, help='Detectron2 output dir')
    parser.add_argument('--config', default='configs/custom/dino_v3_mask2former_wound_instance.yaml')
    parser.add_argument('--dino-weights', default='', help='DINOv3 backbone weights path (optional)')
    parser.add_argument('--model-weights', default='', help='Full model weights path (optional)')
    parser.add_argument('--num-gpus', default='1')
    parser.add_argument('--skip-coco', action='store_true', help='Skip COCO conversion if already done')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    tools_dir = repo_root / 'tools'
    mask2former_root = repo_root.parent / 'Mask2formers'

    base_env = os.environ.copy()
    extra_paths = [str(repo_root)]
    if mask2former_root.exists():
        extra_paths.insert(0, str(mask2former_root))
    base_env['PYTHONPATH'] = os.pathsep.join(extra_paths + [base_env.get('PYTHONPATH', '')]).strip(os.pathsep)

    dataset_root = Path(args.dataset_root)
    coco_out = Path(args.coco_out)
    output_dir = Path(args.output_dir)
    coco_out.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ['WOUND_DATASET_ROOT'] = str(dataset_root)
    os.environ['WOUND_COCO_DIR'] = str(coco_out)

    if not args.skip_coco:
        run([
            sys.executable,
            str(tools_dir / 'create_coco_instances_from_binary_masks.py'),
            '--dataset-root', str(dataset_root),
            '--out-dir', str(coco_out),
            '--category', 'wound',
        ], cwd=str(repo_root), env=base_env)

    train_cmd = [
        sys.executable,
        str(repo_root / 'train_net.py'),
        '--config-file', args.config,
        '--num-gpus', str(args.num_gpus),
        'OUTPUT_DIR', str(output_dir),
    ]

    if args.dino_weights:
        train_cmd += ['MODEL.DINO_V3.WEIGHTS_PATH', args.dino_weights]
    if args.model_weights:
        train_cmd += ['MODEL.WEIGHTS', args.model_weights]

    run(train_cmd, cwd=str(repo_root), env=base_env)


if __name__ == '__main__':
    main()

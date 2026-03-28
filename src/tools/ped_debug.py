import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.getcwd()))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import ViewOfDelft, collate_vod_batch
from src.model.detector import CenterPoint
from src.model.utils import LiDARInstance3DBoxes


def _nested_get(cfg, path, default=None):
    cur = cfg
    for key in path.split('.'):
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
        else:
            return default
    return cur


def _to_plain_dict(obj):
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_dict(v) for v in obj]
    try:
        from omegaconf import OmegaConf  # type: ignore
        if OmegaConf.is_config(obj):
            return _to_plain_dict(OmegaConf.to_container(obj, resolve=True))
    except Exception:
        pass
    return obj


def _parse_thresholds(text: str) -> List[float]:
    values = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        values = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    values = sorted(set(values))
    return values


def _resolve_checkpoint(run_dir: str, checkpoint: str) -> str:
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
        return checkpoint

    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f'Checkpoint directory not found: {ckpt_dir}')

    last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
    if os.path.isfile(last_ckpt):
        return last_ckpt

    candidates = [
        os.path.join(ckpt_dir, name)
        for name in os.listdir(ckpt_dir)
        if name.endswith('.ckpt')
    ]
    if not candidates:
        raise FileNotFoundError(f'No .ckpt files found in: {ckpt_dir}')
    candidates.sort()
    return candidates[-1]


def _load_training_cfg(checkpoint_path: str) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hyper_params = checkpoint.get('hyper_parameters', {})
    cfg = hyper_params.get('config', hyper_params)
    return _to_plain_dict(cfg) if cfg is not None else {}


def _build_val_dataloader(cfg: Dict, num_workers: int) -> DataLoader:
    fusion_enabled = bool(_nested_get(cfg, 'model.fusion.enabled', False))
    if fusion_enabled:
        raise ValueError('Fusion has been removed. Use radar-only checkpoints/configs.')
    camera_rescue_enabled = bool(_nested_get(cfg, 'model.camera_rescue.enabled', False))
    dataset = ViewOfDelft(
        data_root=str(_nested_get(cfg, 'data_root', 'data/view_of_delft')),
        split='val',
        radar_source=str(_nested_get(cfg, 'radar_source', 'radar')),
        radar_prioritize_recent=bool(_nested_get(cfg, 'radar_prioritize_recent', True)),
        radar_z_clip_enabled=bool(_nested_get(cfg, 'radar_z_clip.enabled', False)),
        radar_z_clip_min=float(_nested_get(cfg, 'radar_z_clip.min', -2.5)),
        radar_z_clip_max=float(_nested_get(cfg, 'radar_z_clip.max', 2.0)),
        radar_z_jitter_std=0.0,
        include_camera=camera_rescue_enabled,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(num_workers),
        shuffle=False,
        collate_fn=collate_vod_batch,
    )


def _pairwise_iou_fallback(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> torch.Tensor:
    gt_obj = LiDARInstance3DBoxes(gt_boxes, box_dim=7, origin=(0.5, 0.5, 0))
    pred_obj = LiDARInstance3DBoxes(pred_boxes, box_dim=7, origin=(0.5, 0.5, 0))
    a = gt_obj.nearest_bev
    b = pred_obj.nearest_bev
    if a.shape[0] == 0 or b.shape[0] == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0].unsqueeze(0), b[:, 1].unsqueeze(0), b[:, 2].unsqueeze(0), b[:, 3].unsqueeze(0)
    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter = inter_w * inter_h
    area_a = torch.clamp(ax2 - ax1, min=0.0) * torch.clamp(ay2 - ay1, min=0.0)
    area_b = torch.clamp(bx2 - bx1, min=0.0) * torch.clamp(by2 - by1, min=0.0)
    union = torch.clamp(area_a + area_b - inter, min=1e-6)
    return inter / union


def _pairwise_iou3d(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor, device: torch.device) -> torch.Tensor:
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return torch.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=torch.float32)
    try:
        gt_obj = LiDARInstance3DBoxes(gt_boxes.to(device), box_dim=7, origin=(0.5, 0.5, 0))
        pred_obj = LiDARInstance3DBoxes(pred_boxes.to(device), box_dim=7, origin=(0.5, 0.5, 0))
        iou = LiDARInstance3DBoxes.overlaps(gt_obj, pred_obj)
        return iou.detach().cpu()
    except Exception:
        return _pairwise_iou_fallback(gt_boxes.cpu(), pred_boxes.cpu())


def _greedy_match(
    iou_matrix: torch.Tensor,
    pred_scores: torch.Tensor,
    selected_mask: torch.Tensor,
    iou_threshold: float,
    num_gt: int,
) -> Tuple[int, int, int, torch.Tensor]:
    if num_gt == 0:
        fp = int(selected_mask.sum().item())
        return 0, fp, 0, torch.zeros((0,), dtype=torch.bool)

    selected_idx = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
    if selected_idx.numel() == 0:
        matched_gt = torch.zeros((num_gt,), dtype=torch.bool)
        return 0, 0, num_gt, matched_gt

    selected_scores = pred_scores[selected_idx]
    selected_iou = iou_matrix[:, selected_idx]
    order = torch.argsort(selected_scores, descending=True)

    matched_gt = torch.zeros((num_gt,), dtype=torch.bool)
    tp = 0
    fp = 0
    for pred_rank in order:
        col = selected_iou[:, pred_rank].clone()
        col[matched_gt] = -1.0
        best_iou, best_gt = torch.max(col, dim=0)
        if float(best_iou.item()) >= iou_threshold:
            matched_gt[int(best_gt.item())] = True
            tp += 1
        else:
            fp += 1
    fn = num_gt - tp
    return tp, fp, fn, matched_gt


def _count_points_per_gt_box(points_xyz: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.long)
    if points_xyz.shape[0] == 0:
        return torch.zeros((boxes.shape[0],), dtype=torch.long)

    centers = boxes[:, :3].unsqueeze(1)  # (B,1,3)
    rel = points_xyz.unsqueeze(0) - centers  # (B,P,3)
    # Align world points to each box local axis.
    yaw = boxes[:, 6] + math.pi / 2.0
    c = torch.cos(-yaw).unsqueeze(1)
    s = torch.sin(-yaw).unsqueeze(1)
    local_x = rel[:, :, 0] * c - rel[:, :, 1] * s
    local_y = rel[:, :, 0] * s + rel[:, :, 1] * c
    local_z = rel[:, :, 2]

    half_l = (boxes[:, 3] * 0.5).unsqueeze(1)
    half_w = (boxes[:, 4] * 0.5).unsqueeze(1)
    h = boxes[:, 5].unsqueeze(1)

    inside = (
        (local_x.abs() <= half_l)
        & (local_y.abs() <= half_w)
        & (local_z >= 0.0)
        & (local_z <= h)
    )
    return inside.sum(dim=1).to(torch.long)


def _range_bin(distance_m: float) -> str:
    if distance_m < 15.0:
        return '0-15m'
    if distance_m < 30.0:
        return '15-30m'
    return '30m+'


def _point_bin(num_points: int) -> str:
    if num_points <= 0:
        return '0'
    if num_points <= 2:
        return '1-2'
    if num_points <= 5:
        return '3-5'
    if num_points <= 10:
        return '6-10'
    return '11+'


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Pedestrian-focused debug diagnostics for radar runs.')
    parser.add_argument('--run-dir', type=str, default='outputs/ablate_bev016_dist_dopp_hicap_pfnplus')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--output-dir', type=str, default='')
    parser.add_argument('--score-thresholds', type=str, default='0.03,0.05,0.1,0.2,0.3,0.5')
    parser.add_argument('--eval-score-threshold', type=float, default=0.10)
    parser.add_argument('--iou-threshold', type=float, default=0.25)
    parser.add_argument('--ped-label-index', type=int, default=-1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    start_t = time.time()
    run_dir = os.path.abspath(args.run_dir)
    checkpoint_path = _resolve_checkpoint(run_dir=run_dir, checkpoint=args.checkpoint)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(run_dir, 'ped_debug')
    os.makedirs(output_dir, exist_ok=True)

    thresholds = _parse_thresholds(args.score_thresholds)
    if args.eval_score_threshold not in thresholds:
        thresholds = sorted(set(thresholds + [float(args.eval_score_threshold)]))

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this model path (voxelization uses .cuda()).')
    device = torch.device(args.device)

    cfg = _load_training_cfg(checkpoint_path)
    val_loader = _build_val_dataloader(cfg=cfg, num_workers=args.num_workers)
    model = CenterPoint.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
    model.eval()

    class_names = list(getattr(model, 'class_names', ['Car', 'Pedestrian', 'Cyclist']))
    ped_label = args.ped_label_index
    if ped_label < 0:
        if 'Pedestrian' in class_names:
            ped_label = int(class_names.index('Pedestrian'))
        else:
            ped_label = 1

    threshold_stats = {
        f'{thr:.4f}': {'tp': 0, 'fp': 0, 'fn': 0}
        for thr in thresholds
    }
    range_stats = {k: {'total': 0, 'matched': 0} for k in ['0-15m', '15-30m', '30m+']}
    point_stats = {k: {'total': 0, 'matched': 0} for k in ['0', '1-2', '3-5', '6-10', '11+']}
    sample_failures = []

    num_samples = len(val_loader.dataset)
    print(f'Ped debug run')
    print(f'  checkpoint: {checkpoint_path}')
    print(f'  output_dir: {output_dir}')
    print(f'  samples: {num_samples}')
    print(f'  thresholds: {thresholds}')
    print(f'  iou_threshold: {args.iou_threshold}')
    print(f'  ped_label_index: {ped_label} ({class_names[ped_label] if ped_label < len(class_names) else "unknown"})')

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if args.max_samples > 0 and idx >= args.max_samples:
                break

            ret_dict = model._model_forward(batch)
            bbox_list = model.head.get_bboxes(ret_dict, img_metas=batch['metas'])
            pred_boxes_obj, pred_scores, pred_labels = bbox_list[0]

            gt_boxes = batch['gt_bboxes_3d'][0].tensor[:, :7].detach().cpu()
            gt_labels = batch['gt_labels_3d'][0].detach().cpu()
            points_xyz = batch['pts'][0][:, :3].detach().cpu()

            ped_gt_mask = (gt_labels == ped_label)
            ped_gt_boxes = gt_boxes[ped_gt_mask]
            ped_pred_mask = (pred_labels.detach().cpu() == ped_label)
            pred_boxes = pred_boxes_obj.tensor[:, :7].detach().cpu()
            ped_pred_boxes = pred_boxes[ped_pred_mask]
            ped_pred_scores = pred_scores.detach().cpu()[ped_pred_mask]

            iou_matrix = _pairwise_iou3d(ped_gt_boxes, ped_pred_boxes, device=device)
            n_gt = int(ped_gt_boxes.shape[0])

            eval_matched = torch.zeros((n_gt,), dtype=torch.bool)
            for thr in thresholds:
                key = f'{thr:.4f}'
                selected_mask = (ped_pred_scores >= thr)
                tp, fp, fn, matched_gt = _greedy_match(
                    iou_matrix=iou_matrix,
                    pred_scores=ped_pred_scores,
                    selected_mask=selected_mask,
                    iou_threshold=args.iou_threshold,
                    num_gt=n_gt,
                )
                threshold_stats[key]['tp'] += int(tp)
                threshold_stats[key]['fp'] += int(fp)
                threshold_stats[key]['fn'] += int(fn)
                if abs(thr - args.eval_score_threshold) < 1e-12:
                    eval_matched = matched_gt.clone()

            if n_gt > 0:
                gt_distances = torch.linalg.norm(ped_gt_boxes[:, :2], ord=2, dim=1)
                gt_point_counts = _count_points_per_gt_box(points_xyz=points_xyz, boxes=ped_gt_boxes)
                for g in range(n_gt):
                    range_key = _range_bin(float(gt_distances[g].item()))
                    point_key = _point_bin(int(gt_point_counts[g].item()))
                    range_stats[range_key]['total'] += 1
                    point_stats[point_key]['total'] += 1
                    if bool(eval_matched[g].item()):
                        range_stats[range_key]['matched'] += 1
                        point_stats[point_key]['matched'] += 1

                sample_idx = str(batch['metas'][0].get('num_frame', f'idx_{idx}'))
                missed = int((~eval_matched).sum().item())
                if missed > 0:
                    sample_failures.append({
                        'sample_idx': sample_idx,
                        'ped_gt': n_gt,
                        'ped_missed': missed,
                    })

            if (idx + 1) % 200 == 0:
                print(f'  processed {idx + 1}/{num_samples}')

    threshold_rows = []
    for thr in thresholds:
        key = f'{thr:.4f}'
        tp = threshold_stats[key]['tp']
        fp = threshold_stats[key]['fp']
        fn = threshold_stats[key]['fn']
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        threshold_rows.append({
            'score_threshold': float(thr),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

    range_rows = []
    for key in ['0-15m', '15-30m', '30m+']:
        total = range_stats[key]['total']
        matched = range_stats[key]['matched']
        range_rows.append({
            'range_bin': key,
            'gt_total': int(total),
            'gt_matched': int(matched),
            'recall': _safe_ratio(matched, total),
        })

    point_rows = []
    for key in ['0', '1-2', '3-5', '6-10', '11+']:
        total = point_stats[key]['total']
        matched = point_stats[key]['matched']
        point_rows.append({
            'points_per_gt_bin': key,
            'gt_total': int(total),
            'gt_matched': int(matched),
            'recall': _safe_ratio(matched, total),
        })

    sample_failures.sort(key=lambda x: (x['ped_missed'], x['ped_gt']), reverse=True)
    sample_failures = sample_failures[:100]

    elapsed = time.time() - start_t
    summary = {
        'checkpoint_path': checkpoint_path,
        'run_dir': run_dir,
        'ped_label_index': ped_label,
        'class_names': class_names,
        'iou_threshold': float(args.iou_threshold),
        'eval_score_threshold': float(args.eval_score_threshold),
        'threshold_sweep': threshold_rows,
        'range_bins': range_rows,
        'point_bins': point_rows,
        'top_failed_samples': sample_failures,
        'num_val_samples_processed': int(args.max_samples) if args.max_samples > 0 else int(num_samples),
        'runtime_sec': float(elapsed),
    }

    summary_path = os.path.join(output_dir, 'ped_debug_summary.json')
    thresholds_csv = os.path.join(output_dir, 'ped_threshold_sweep.csv')
    range_csv = os.path.join(output_dir, 'ped_range_bins.csv')
    points_csv = os.path.join(output_dir, 'ped_points_bins.csv')
    fails_csv = os.path.join(output_dir, 'ped_top_failed_samples.csv')

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    _write_csv(thresholds_csv, threshold_rows, ['score_threshold', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])
    _write_csv(range_csv, range_rows, ['range_bin', 'gt_total', 'gt_matched', 'recall'])
    _write_csv(points_csv, point_rows, ['points_per_gt_bin', 'gt_total', 'gt_matched', 'recall'])
    _write_csv(fails_csv, sample_failures, ['sample_idx', 'ped_gt', 'ped_missed'])

    best_row = max(threshold_rows, key=lambda r: (r['recall'], r['f1']))
    eval_row = min(threshold_rows, key=lambda r: abs(r['score_threshold'] - args.eval_score_threshold))
    print('\nDone.')
    print(f'  best recall threshold: {best_row["score_threshold"]:.4f} | '
          f'precision={best_row["precision"]:.4f} recall={best_row["recall"]:.4f} f1={best_row["f1"]:.4f}')
    print(f'  eval threshold: {eval_row["score_threshold"]:.4f} | '
          f'precision={eval_row["precision"]:.4f} recall={eval_row["recall"]:.4f} f1={eval_row["f1"]:.4f}')
    print(f'  wrote: {summary_path}')
    print(f'         {thresholds_csv}')
    print(f'         {range_csv}')
    print(f'         {points_csv}')
    print(f'         {fails_csv}')


if __name__ == '__main__':
    main()

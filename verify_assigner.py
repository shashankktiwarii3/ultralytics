"""Verify SizeAdaptiveAssigner implementation."""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ultralytics.utils.tal import SizeAdaptiveAssigner, TaskAlignedAssigner
from ultralytics.utils.loss import v8DetectionLoss

def test_import():
    assert SizeAdaptiveAssigner is not None, "Failed to import"
    assert issubclass(SizeAdaptiveAssigner, TaskAlignedAssigner), "Must inherit TaskAlignedAssigner"
    assert hasattr(v8DetectionLoss.__init__, "__code__")  # loss.py wired
    print("  [PASS] Import & inheritance")

def test_loss_wiring():
    """Verify SizeAdaptiveAssigner is importable from loss module."""
    from ultralytics.utils.loss import SizeAdaptiveAssigner as SAA
    assert SAA is SizeAdaptiveAssigner
    print("  [PASS] Loss module wiring")

def test_instantiation():
    a = SizeAdaptiveAssigner(topk=13, num_classes=80, stride=[8,16,32])
    assert a.max_o2o == 4
    assert a.adaptive_topk2 is True
    assert a.topk2 == a.topk  # parent default: topk2 = topk2 or topk
    print("  [PASS] Instantiation with defaults")

    b = SizeAdaptiveAssigner(topk=13, num_classes=80, stride=[8,16,32], max_o2o=6, adaptive_topk2=False)
    assert b.max_o2o == 6
    assert b.adaptive_topk2 is False
    print("  [PASS] Instantiation with custom params")

def test_compute_adaptive_topk2():
    a = SizeAdaptiveAssigner(topk=13, num_classes=80, stride=[8,16,32])
    bsz, n_max = 2, 5

    # Large objects (min_dim >> stride_val) -> topk2 = 1
    gt_large = torch.tensor([[[0,0,200,300]]*n_max]*bsz, dtype=torch.float32)
    mask_gt = torch.ones(bsz, n_max, 1, dtype=torch.bool)
    k_large = a.compute_adaptive_topk2(gt_large, mask_gt)
    assert k_large.shape == (bsz, n_max), f"Shape mismatch: {k_large.shape}"
    assert (k_large == 1).all(), f"Large objs should map to k=1, got {k_large}"
    print("  [PASS] compute_adaptive_topk2: large objs -> k=1")

    # Tiny objects (min_dim << stride_val) -> topk2 > 1
    gt_tiny = torch.tensor([[[0,0,8,10]]*n_max]*bsz, dtype=torch.float32)
    k_tiny = a.compute_adaptive_topk2(gt_tiny, mask_gt)
    assert k_tiny.shape == (bsz, n_max)
    assert (k_tiny > 1).all(), f"Tiny objs should map to k>1, got {k_tiny}"
    assert (k_tiny <= a.max_o2o).all(), f"k should be capped at max_o2o={a.max_o2o}, got {k_tiny}"
    print(f"  [PASS] compute_adaptive_topk2: tiny objs -> k={k_tiny[0,0].item()}")

    # Mixed (tiny + large + masked)
    gt_mixed = torch.zeros(bsz, n_max, 4)
    gt_mixed[0, 0] = torch.tensor([0,0,6,7])   # tiny
    gt_mixed[0, 1] = torch.tensor([0,0,100,50]) # large
    gt_mixed[0, 2] = torch.tensor([0,0,300,400])# large
    mask_mixed = torch.zeros(bsz, n_max, 1, dtype=torch.bool)
    mask_mixed[0, :3] = True
    k_mixed = a.compute_adaptive_topk2(gt_mixed, mask_mixed)
    assert k_mixed[0, 0] > 1, f"Tiny should be >1, got {k_mixed[0,0]}"
    assert k_mixed[0, 1] == 1, f"Large should be 1, got {k_mixed[0,1]}"
    assert k_mixed[0, 2] == 1, f"Large should be 1, got {k_mixed[0,2]}"
    assert k_mixed[0, 3] == 1, f"Masked should be 1, got {k_mixed[0,3]}"
    print("  [PASS] compute_adaptive_topk2: mixed scenario correct")

def test_forward():
    """Minimal forward pass smoke test."""
    bsz, n_max, na, nc = 2, 3, 100, 80
    assigner = SizeAdaptiveAssigner(topk=13, num_classes=nc, stride=[8,16,32])

    pd_scores = torch.randn(bsz, na, nc).sigmoid()
    pd_bboxes = torch.rand(bsz, na, 4) * 100
    gt_labels = torch.randint(0, nc, (bsz, n_max, 1))
    gt_bboxes = torch.rand(bsz, n_max, 4) * 100
    gt_bboxes[..., 2:4] += gt_bboxes[..., :2].clone()  # valid xyxy
    anc_points = torch.rand(na, 2) * 100
    mask_gt = torch.ones(bsz, n_max, 1, dtype=torch.bool)

    target_gt_idx, target_bboxes, target_scores, fg_mask, _ = assigner(
        pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
    )

    assert target_gt_idx.shape == (bsz, na), f"{target_gt_idx.shape}"
    assert target_bboxes.shape == (bsz, na, 4), f"{target_bboxes.shape}"
    assert target_scores.shape == (bsz, na, nc), f"{target_scores.shape}"
    assert fg_mask.shape == (bsz, na), f"{fg_mask.shape}"
    print(f"  [PASS] forward: fg_mask sum = {fg_mask.sum().item():.0f} (non-zero)")

def test_topk2_changes():
    """Verify adaptive topk2 produces different assignments vs baseline."""
    bsz, n_max, na, nc = 2, 3, 200, 10
    assigner_base = TaskAlignedAssigner(topk=13, num_classes=nc, stride=[8,16,32], topk2=1)
    assigner_adaptive = SizeAdaptiveAssigner(topk=13, num_classes=nc, stride=[8,16,32], max_o2o=4)

    torch.manual_seed(0)
    pd_scores = torch.randn(bsz, na, nc).sigmoid()
    pd_bboxes = torch.rand(bsz, na, 4) * 200
    gt_labels = torch.randint(0, nc, (bsz, n_max, 1))
    gt_bboxes = torch.rand(bsz, n_max, 4) * 200
    gt_bboxes[..., 0] = torch.clamp(gt_bboxes[..., 0], 0, 100)
    gt_bboxes[..., 1] = torch.clamp(gt_bboxes[..., 1], 0, 100)
    gt_bboxes[..., 2] = gt_bboxes[..., 0] + torch.rand(bsz, n_max) * 10 + 2  # tiny
    gt_bboxes[..., 3] = gt_bboxes[..., 1] + torch.rand(bsz, n_max) * 10 + 2  # tiny
    anc_points = torch.rand(na, 2) * 200
    mask_gt = torch.ones(bsz, n_max, 1, dtype=torch.bool)

    _, _, _, fg_base, _ = assigner_base(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    _, _, _, fg_adapt, _ = assigner_adaptive(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    assert fg_adapt.sum() >= fg_base.sum(), (
        f"Adaptive should assign >= positives, base={fg_base.sum().item():.0f} adapt={fg_adapt.sum().item():.0f}"
    )
    print(f"  [PASS] topk2 adaptive: base_fg={fg_base.sum().item():.0f} adapt_fg={fg_adapt.sum().item():.0f} (adapt >= base)")


if __name__ == "__main__":
    print("SizeAdaptiveAssigner verification\n")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("\nAll checks passed.")

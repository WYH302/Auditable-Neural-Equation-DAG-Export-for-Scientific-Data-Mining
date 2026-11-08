# V2 Redo Notes

Current status:

- `eml_kan_tau15_ood`: two-layer EML-KAN, tau 1.5, OOD-aware early stopping.
- `eml_kan_tau20_ood`: two-layer EML-KAN, tau 2.0, OOD-aware early stopping.
- `eml_kan_depth2_ood`: EML-KAN with edge depth 2, tau 2.0, OOD weight 0.10.
- `eml_kan_depth4_ood`: EML-KAN with edge depth 4, tau 2.0, OOD weight 0.10.
- `eml_kan_depth2_oodw025`: edge depth 2, OOD weight 0.25.
- `eml_kan_depth2_oodw050`: edge depth 2, OOD weight 0.50.
- `eml_edge_lr_probe`: one-layer EML edge probe with higher learning rate.

Preliminary takeaway:

- All v2 runs are numerically stable so far.
- OOD-aware early stopping improves some cases, especially compared with the
  first minimal Stable-EML network, but it does not fully solve OOD.
- One-layer `eml_edge` is not enough; it underfits or extrapolates poorly on
  exp/exp-log probes.
- Increasing OOD early-stop weight from 0.10 to 0.25/0.50 did not clearly
  improve the best OOD means in the current core synthetic sweep.
- Edge depth 2 is the safest current default for OOD among the tried EML-KAN
  variants; depth 4 can improve interpolation but is not consistently better
  out of domain.

Next redo direction:

1. Keep `eml_kan` as the main v2 architecture.
2. Add explicit symbolic/structural extraction instead of relying on neural OOD.
3. Add an ablation for `tau`, edge depth, and OOD weight.
4. Treat current v2 runs as calibration, not final paper tables.

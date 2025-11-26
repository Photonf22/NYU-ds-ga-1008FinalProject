Swap backbones in training to create baseline configurations:
SwinT, Convnext, Vit_b_16

|                  | ViT_b_16 | ConvNext Tiny | ConvNext Small | ConvNext Base | Devel | Swin Small | Swin Tiny |
|------------------|----------|---------------|----------------|---------------|-------|------------|-----------|
| # of Params (M)  |  70.9    |     99.5      |      70.8      |               |       |  49.7      |   28.3    |
| Top 5 Acc        |          |               |                |               |       |            |           |
| FLOPs per img    |          |               |                |               |       |            |           |
| Acc per FLOP     |          |               |                |               |       |            |           |
| Peak Activation  |          |               |                |               |       |            |           |
| Robustness (OOD) |          |               |                |               |       |            |           

- **Description:** Compare SwinT, ConvNeXt, ViT, and iJEPA/DINO baselines with parameter/FLOP budgets and downstream metrics.
- **Meeting notes to apply:** Prioritize Vision Transformers and DINO/J(E)PA-style backbones; MOE is too complex for the current <100M parameter target.
- **Research/foundations:** Characterization should report parameter count, FLOPs, throughput, and linear-probe/fine-tune accuracy to reveal representation quality. Track attention map interpretability and masking/predictor design choices that influence stability in predictive SSL.
- **Next actions:** Build a comparison table covering model size, input resolution, FLOPs, and evaluation results (linear probe, k-NN, full fine-tune); include a short narrative on why iJEPA/DINO are favored within the budget.

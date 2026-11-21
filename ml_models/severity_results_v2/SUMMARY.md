# Results summary (v2)
mode: full, images: 5837, classes: 15, seeds: [42, 7, 2026], epochs: 8
auto-excluded classes: ['Apple___Apple_scab', 'Tomato___Bacterial_spot', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot']

## Label check
- false-positive symptomatic % on healthy leaves (mean): 0.89
- within-class share of severity variance: 0.593

## Test results (mean ± std over seeds)
                               acc       macro_f1            qwk qwk_diseased_only within_class_spearman        mae_pct    pearson_pct        mae_nec        mae_chl qwk_grade_head
class_only_baseline          0.680          0.669          0.771             0.428                 0.000          8.513          0.658              –              –              –
resnet50_ce          0.899 ± 0.006  0.897 ± 0.006  0.953 ± 0.003     0.874 ± 0.007         0.875 ± 0.011              –              –              –              –  0.953 ± 0.003
vit_ce               0.917 ± 0.011  0.916 ± 0.012  0.963 ± 0.005     0.902 ± 0.012         0.910 ± 0.012              –              –              –              –  0.963 ± 0.005
resnet50_ce_reg      0.910 ± 0.005  0.909 ± 0.005  0.960 ± 0.003     0.893 ± 0.007         0.910 ± 0.002  2.132 ± 0.081  0.976 ± 0.001  1.445 ± 0.089  1.584 ± 0.094  0.959 ± 0.001
vit_coral_reg        0.928 ± 0.001  0.927 ± 0.001  0.966 ± 0.000     0.911 ± 0.002         0.944 ± 0.000  1.519 ± 0.016  0.988 ± 0.000  1.169 ± 0.039  1.088 ± 0.018  0.950 ± 0.002
PROPOSED_vit_ce_reg  0.941 ± 0.003  0.940 ± 0.003  0.973 ± 0.001     0.927 ± 0.003         0.943 ± 0.003  1.601 ± 0.018  0.987 ± 0.001  1.245 ± 0.057  1.191 ± 0.047  0.969 ± 0.002
vit_ce_reg_dann      0.934 ± 0.002  0.933 ± 0.002  0.970 ± 0.001     0.919 ± 0.003         0.937 ± 0.005  1.726 ± 0.046  0.985 ± 0.000  1.339 ± 0.042  1.310 ± 0.036  0.967 ± 0.002

## Bootstrap (QWK difference, proposed − other)
- seed42 QWK proposed - vit_ce: +0.003 [-0.003, +0.009]
- seed42 QWK proposed - resnet50_ce: +0.022 [+0.012, +0.033]
- seed42 QWK proposed - resnet50_ce_reg: +0.009 [+0.001, +0.017]
- seed42 QWK proposed - vit_coral_reg: +0.005 [-0.001, +0.012]
- seed7 QWK proposed - vit_ce: +0.016 [+0.010, +0.023]
- seed7 QWK proposed - resnet50_ce: +0.020 [+0.012, +0.029]
- seed7 QWK proposed - resnet50_ce_reg: +0.017 [+0.009, +0.025]
- seed7 QWK proposed - vit_coral_reg: +0.007 [+0.000, +0.014]
- seed2026 QWK proposed - vit_ce: +0.010 [+0.003, +0.017]
- seed2026 QWK proposed - resnet50_ce: +0.018 [+0.009, +0.029]
- seed2026 QWK proposed - resnet50_ce_reg: +0.013 [+0.005, +0.022]
- seed2026 QWK proposed - vit_coral_reg: +0.007 [-0.000, +0.014]

## Explainability
{"attention_in_lesion_ratio_median": 1.116913348789193, "frac_images_ratio_gt1": 0.96, "n": 200}
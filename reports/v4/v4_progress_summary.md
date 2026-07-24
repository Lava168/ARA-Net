# V4 AD Rescue Progress Summary

## Old v3 external baseline

- aibl: n=1307, Acc=0.606, BAcc=0.399, AUC=0.597, pred={'CN': 918, 'MCI': 44, 'AD': 345}
- ixi: n=581, Acc=0.439, BAcc=0.439, AUC=NA, CN_retention=0.439, pred={'CN': 255, 'MCI': 41, 'AD': 285}
- oasis: n=99, Acc=0.606, BAcc=0.364, AUC=0.663, pred={'CN': 97, 'MCI': 0, 'AD': 2}

## Atlas feature baseline

Best model: `hgb` score=0.5273
- val: n=355, Acc=0.546, BAcc=0.492, AUC=0.731, ADvCN_AUC=0.925, pred={'CN': 159, 'MCI': 161, 'AD': 35}; recall(CN=0.744, MCI=0.530, AD=0.203)
- internal_test: n=360, Acc=0.561, BAcc=0.516, AUC=0.720, ADvCN_AUC=0.884, pred={'CN': 164, 'MCI': 167, 'AD': 29}; recall(CN=0.811, MCI=0.585, AD=0.154)
- aibl_adapt_val: n=191, Acc=0.817, BAcc=0.552, AUC=0.770, ADvCN_AUC=0.975, pred={'CN': 171, 'MCI': 4, 'AD': 16}; recall(CN=0.973, MCI=0.000, AD=0.684)
- aibl_heldout: n=397, Acc=0.776, BAcc=0.479, AUC=0.732, ADvCN_AUC=0.884, pred={'CN': 346, 'MCI': 18, 'AD': 33}; recall(CN=0.944, MCI=0.151, AD=0.341)
- oasis_external: n=99, Acc=0.192, BAcc=0.310, AUC=0.523, ADvCN_AUC=0.432, pred={'CN': 16, 'MCI': 5, 'AD': 78}; recall(CN=0.169, MCI=0.034, AD=0.727)
- ixi_external: n=581, Acc=0.983, BAcc=0.983, AUC=NA, CN_retention=0.983, pred={'CN': 571, 'MCI': 6, 'AD': 4}; recall(CN=0.983, MCI=0.000, AD=0.000)

## Atlas biomarker validation

- aibl_heldout: AD-key volume score=0.510, uniform=0.286, delta=0.225, CI=[0.479, 0.526], p=0.0260
- aibl_adapt_heldout: AD-key volume score=0.512, uniform=0.286, delta=0.226, CI=[0.493, 0.525], p=0.0350
- all_labeled_ad: AD-key volume score=0.426, uniform=0.286, delta=0.141, CI=[0.348, 0.478], p=0.0207
- adni_val_internal_test: AD-key volume score=0.342, uniform=0.286, delta=0.056, CI=[0.258, 0.503], p=0.1843

## Deep v4 status

- aranet_v4_aibl_adapt_corrected_selection_seed42: epochs=7, best_epoch=5, best_score=0.5035
  - val: n=546, Acc=0.564, BAcc=0.457, AUC=0.628, ADvCN_AUC=0.639, pred={'CN': 215, 'MCI': 331, 'AD': 0}; recall(CN=0.566, MCI=0.806, AD=0.000)
  - aibl_heldout: n=397, Acc=0.763, BAcc=0.333, AUC=0.469, ADvCN_AUC=0.415, pred={'CN': 397, 'MCI': 0, 'AD': 0}; recall(CN=1.000, MCI=0.000, AD=0.000)
  - ixi_external: n=581, Acc=0.898, BAcc=0.898, AUC=NA, CN_retention=0.898, pred={'CN': 522, 'MCI': 59, 'AD': 0}; recall(CN=0.898, MCI=0.000, AD=0.000)
- aranet_v4_aibl_adapt_prior_corrected_seed42: epochs=8, best_epoch=6, best_score=0.6196
  - val: n=546, Acc=0.474, BAcc=0.488, AUC=0.692, ADvCN_AUC=0.770, pred={'CN': 194, 'MCI': 172, 'AD': 180}; recall(CN=0.518, MCI=0.366, AD=0.578)
  - aibl_heldout: n=397, Acc=0.650, BAcc=0.347, AUC=0.592, ADvCN_AUC=0.679, pred={'CN': 324, 'MCI': 1, 'AD': 72}; recall(CN=0.822, MCI=0.000, AD=0.220)
  - ixi_external: n=581, Acc=1.000, BAcc=1.000, AUC=NA, CN_retention=1.000, pred={'CN': 581, 'MCI': 0, 'AD': 0}; recall(CN=1.000, MCI=0.000, AD=0.000)
- aranet_v4_aibl_adapt_seed42: epochs=13, best_epoch=3, best_score=0.4499
  - val: n=546, Acc=0.570, BAcc=0.471, AUC=0.639, ADvCN_AUC=0.625, pred={'CN': 191, 'MCI': 349, 'AD': 6}; recall(CN=0.540, MCI=0.848, AD=0.024)
  - aibl_heldout: n=397, Acc=0.763, BAcc=0.333, AUC=0.451, ADvCN_AUC=0.367, pred={'CN': 397, 'MCI': 0, 'AD': 0}; recall(CN=1.000, MCI=0.000, AD=0.000)
  - ixi_external: n=581, Acc=0.210, BAcc=0.210, AUC=NA, CN_retention=0.210, pred={'CN': 122, 'MCI': 459, 'AD': 0}; recall(CN=0.210, MCI=0.000, AD=0.000)
- aranet_v4_aibl_adapt_specificity_seed42: epochs=10, best_epoch=8, best_score=0.4290
  - val: n=546, Acc=0.445, BAcc=0.404, AUC=0.615, ADvCN_AUC=0.646, pred={'CN': 325, 'MCI': 55, 'AD': 166}; recall(CN=0.688, MCI=0.115, AD=0.410)
  - aibl_heldout: n=397, Acc=0.763, BAcc=0.333, AUC=0.424, ADvCN_AUC=0.353, pred={'CN': 397, 'MCI': 0, 'AD': 0}; recall(CN=1.000, MCI=0.000, AD=0.000)
  - ixi_external: n=581, Acc=0.938, BAcc=0.938, AUC=NA, CN_retention=0.938, pred={'CN': 545, 'MCI': 36, 'AD': 0}; recall(CN=0.938, MCI=0.000, AD=0.000)
- aranet_v4_zeroshot_seed42: epochs=15, best_epoch=8, best_score=0.5584
  - val: n=355, Acc=0.372, BAcc=0.412, AUC=0.643, ADvCN_AUC=0.823, pred={'CN': 0, 'MCI': 203, 'AD': 152}; recall(CN=0.000, MCI=0.518, AD=0.719)
  - aibl_heldout: n=397, Acc=0.131, BAcc=0.391, AUC=0.583, ADvCN_AUC=0.696, pred={'CN': 0, 'MCI': 172, 'AD': 225}; recall(CN=0.000, MCI=0.321, AD=0.854)
  - ixi_external: n=581, Acc=0.000, BAcc=0.000, AUC=NA, CN_retention=0.000, pred={'CN': 0, 'MCI': 581, 'AD': 0}; recall(CN=0.000, MCI=0.000, AD=0.000)

## Atlas cascade baseline

Best model: `rf__logreg` score=0.5461
- val: n=355, Acc=0.513, BAcc=0.533, AUC=0.686, ADvCN_AUC=0.931, pred={'CN': 172, 'MCI': 93, 'AD': 90}; recall(CN=0.824, MCI=0.289, AD=0.484)
- aibl_adapt_val: n=191, Acc=0.838, BAcc=0.572, AUC=0.763, ADvCN_AUC=0.960, pred={'CN': 175, 'MCI': 1, 'AD': 15}; recall(CN=0.993, MCI=0.040, AD=0.684)
- aibl_heldout: n=397, Acc=0.751, BAcc=0.391, AUC=0.756, ADvCN_AUC=0.886, pred={'CN': 361, 'MCI': 5, 'AD': 31}; recall(CN=0.954, MCI=0.000, AD=0.220)
- oasis_external: n=99, Acc=0.545, BAcc=0.305, AUC=0.461, ADvCN_AUC=0.598, pred={'CN': 84, 'MCI': 4, 'AD': 11}; recall(CN=0.915, MCI=0.000, AD=0.000)
- ixi_external: n=581, Acc=1.000, BAcc=1.000, AUC=NA, CN_retention=1.000, pred={'CN': 581, 'MCI': 0, 'AD': 0}; recall(CN=1.000, MCI=0.000, AD=0.000)

## Hybrid atlas + clinical baseline

Protocol `adni_only` best: `atlas_biomarker_enhanced__rf_balanced` score=0.5742
- val: n=355, Acc=0.589, BAcc=0.535, AUC=0.792, ADvCN_AUC=0.978, pred={'CN': 130, 'MCI': 185, 'AD': 40}; recall(CN=0.704, MCI=0.620, AD=0.281)
- internal_test: n=360, Acc=0.636, BAcc=0.580, AUC=0.793, ADvCN_AUC=0.937, pred={'CN': 111, 'MCI': 214, 'AD': 35}; recall(CN=0.703, MCI=0.754, AD=0.282)
- aibl_adapt_val: n=191, Acc=0.571, BAcc=0.424, AUC=0.755, ADvCN_AUC=0.958, pred={'CN': 102, 'MCI': 89, 'AD': 0}; recall(CN=0.633, MCI=0.640, AD=0.000)
- aibl_heldout: n=397, Acc=0.549, BAcc=0.406, AUC=0.753, ADvCN_AUC=0.870, pred={'CN': 210, 'MCI': 187, 'AD': 0}; recall(CN=0.614, MCI=0.604, AD=0.000)
- oasis_external: n=99, Acc=0.192, BAcc=0.357, AUC=0.575, ADvCN_AUC=0.599, pred={'CN': 2, 'MCI': 23, 'AD': 74}; recall(CN=0.034, MCI=0.310, AD=0.727)
- ixi_external: n=581, Acc=0.179, BAcc=0.179, AUC=NA, CN_retention=0.179, pred={'CN': 104, 'MCI': 477, 'AD': 0}; recall(CN=0.179, MCI=0.000, AD=0.000)

Protocol `aibl_adapted` best: `atlas_biomarker_enhanced__hgb` score=0.7307
- val: n=355, Acc=0.555, BAcc=0.508, AUC=0.757, ADvCN_AUC=0.965, pred={'CN': 137, 'MCI': 170, 'AD': 48}; recall(CN=0.704, MCI=0.554, AD=0.266)
- internal_test: n=360, Acc=0.625, BAcc=0.570, AUC=0.791, ADvCN_AUC=0.947, pred={'CN': 120, 'MCI': 199, 'AD': 41}; recall(CN=0.784, MCI=0.708, AD=0.218)
- aibl_adapt_val: n=191, Acc=0.911, BAcc=0.836, AUC=0.938, ADvCN_AUC=0.999, pred={'CN': 151, 'MCI': 21, 'AD': 19}; recall(CN=0.959, MCI=0.600, AD=0.947)
- aibl_heldout: n=397, Acc=0.861, BAcc=0.703, AUC=0.942, ADvCN_AUC=0.990, pred={'CN': 314, 'MCI': 40, 'AD': 43}; recall(CN=0.957, MCI=0.396, AD=0.756)
- oasis_external: n=99, Acc=0.192, BAcc=0.310, AUC=0.561, ADvCN_AUC=0.443, pred={'CN': 18, 'MCI': 4, 'AD': 77}; recall(CN=0.169, MCI=0.034, AD=0.727)
- ixi_external: n=581, Acc=0.997, BAcc=0.997, AUC=NA, CN_retention=0.997, pred={'CN': 579, 'MCI': 2, 'AD': 0}; recall(CN=0.997, MCI=0.000, AD=0.000)


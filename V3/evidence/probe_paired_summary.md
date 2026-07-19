# Probe paired bootstrap comparisons

> Each row uses 10,000 paired bootstrap resamples over `structure_id` clusters.
> These are per-seed development comparisons, not a substitute for 4+ seed confirmatory replication.

| baseline | candidate | panel | images | units | exact delta | 95% CI | valid delta | gate |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| data_11_s1 | aug_dose2_s1 | legacy_core_dev | 753 | 743 | -0.001346 | [-0.012786, 0.010111] | -0.005384 | False |
| data_11_s1 | aug_dose2_s1 | legacy_region_dev | 754 | 744 | -0.002688 | [-0.014785, 0.009409] | -0.004032 | False |
| data_00_s1 | data_01_s1 | legacy_core_dev | 753 | 743 | 0.000673 | [-0.010767, 0.012113] | 0.033647 | False |
| data_00_s1 | data_01_s1 | legacy_region_dev | 754 | 744 | 0.000672 | [-0.011425, 0.012769] | 0.037634 | False |
| data_00_s2 | data_01_s2 | legacy_core_dev | 753 | 743 | -0.009421 | [-0.022880, 0.003382] | -0.019515 | False |
| data_00_s2 | data_01_s2 | legacy_region_dev | 754 | 744 | -0.012097 | [-0.025538, 0.001344] | -0.012769 | False |
| data_00_s1 | data_10_s1 | legacy_core_dev | 753 | 743 | 0.004711 | [-0.004711, 0.014805] | 0.037012 | True |
| data_00_s1 | data_10_s1 | legacy_region_dev | 754 | 744 | 0.002016 | [-0.008065, 0.013441] | 0.042339 | False |
| data_00_s2 | data_10_s2 | legacy_core_dev | 753 | 743 | -0.018170 | [-0.030956, -0.006057] | -0.014805 | False |
| data_00_s2 | data_10_s2 | legacy_region_dev | 754 | 744 | -0.019489 | [-0.032258, -0.007392] | -0.006720 | False |
| data_00_s1 | data_11_s1 | legacy_core_dev | 753 | 743 | 0.008748 | [-0.003365, 0.021534] | 0.024899 | True |
| data_00_s1 | data_11_s1 | legacy_region_dev | 754 | 744 | 0.006048 | [-0.006720, 0.019489] | 0.031586 | False |
| warmstart_control_s1 | data_11_s1 | legacy_core_dev | 753 | 743 | 0.337820 | [0.304172, 0.371467] | 0.452894 | True |
| warmstart_control_s1 | data_11_s1 | legacy_region_dev | 754 | 744 | 0.346774 | [0.313172, 0.380376] | 0.506048 | True |
| data_00_s2 | data_11_s2 | legacy_core_dev | 753 | 743 | -0.006057 | [-0.018170, 0.006057] | -0.008748 | False |
| data_00_s2 | data_11_s2 | legacy_region_dev | 754 | 744 | -0.011425 | [-0.023522, 0.000672] | -0.006048 | False |

A CI crossing zero means the observed per-seed difference is compatible with both improvement and regression at this development-sample resolution.

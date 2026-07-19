# V3 release-readiness audit

| manifest | rows | license coverage | source URL coverage | structure ID coverage | QC status |
| --- | ---: | ---: | ---: | ---: | --- |
| final_train_control | 22762 | 0.0% | 0.0% | 0.0% | `{"missing": 22762}` |
| dev_legacy_core | 753 | 100.0% | 100.0% | 100.0% | `{"pass": 753}` |
| dev_legacy_region | 754 | 100.0% | 100.0% | 100.0% | `{"pass": 754}` |
| wild_strict_locked | 301 | 100.0% | 100.0% | 100.0% | `{"pending_manual_review": 301}` |
| wild_symbolic_locked | 460 | 100.0% | 100.0% | 100.0% | `{"pending_manual_review": 460}` |

## Project files

- LICENSE: present
- NOTICE: present
- data_license_matrix: present
- CONTRIBUTING: present
- Dockerfile: missing
- model_card: present
- dataset_card: present
- reproduction_guide: present

## Human and private evaluation status

- Manual-review sheet rows: 1068
- Pending final decisions: 1068
- Review panels: `{"core_767": 767, "wild_strict_v3": 301}`
- Owner attestation: complete
- Private-photo rows: 0

## Release blockers

- final training manifest lacks complete sample-level license and source URL fields
- private-photo evaluation set is empty
- no container image or Dockerfile has been independently reproduced

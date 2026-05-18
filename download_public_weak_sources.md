# Public Weak-Domain Sources Download Guide

This guide fixes four things for each public source:

- official download entry
- what part is worth using for this project
- where it should live under `V2-1/`
- what we should do with it after download

## Directory Plan

Use this local layout:

```text
V2-1/data/public_sources/
  decimer_handdrawn/
  patcid/
  pubchem/
  chembl/
  manifests/
```

Keep original downloaded files in these folders. Do not mix them into `train` or `eval` directly.

## 1. DECIMER Hand-Drawn Dataset

Official entry points:

- Zenodo: [DECIMER Hand-drawn molecule images](https://zenodo.org/records/7617107)
- Project / code context: [DECIMER on GitHub](https://github.com/Kohulan/DECIMER-Image_Transformer)

Why we want it:

- Best public match for the current `decimer` weak domain.
- Good for both training and a held-out handwritten benchmark split.

Where to store:

```text
V2-1/data/public_sources/decimer_handdrawn/
```

Recommended landing layout:

```text
V2-1/data/public_sources/decimer_handdrawn/
  raw/
  manifests/
  README_source.md
```

What to extract:

- images
- any provided labels or source tables
- license / citation info

How to use:

- First build a manifest of image path + SMILES.
- Hold out a small portion for evaluation if needed.
- Send the rest through:

```bash
python V2-1/scripts/prepare_weak_domain_manifest.py ...
python V2-1/scripts/import_weak_domain_training_pool.py ...
```

Recommendation:

- Training: yes
- Evaluation: yes, but only a held-out subset not used in training

## 2. PatCID

Official entry points:

- Paper: [PatCID: an open-access, large-scale corpus of chemical structures in patent documents](https://www.nature.com/articles/s41467-024-50779-y)
- Zenodo: [PatCID dataset](https://zenodo.org/records/10572870)
- GitHub: [DS4SD/PatCID](https://github.com/DS4SD/PatCID)

Why we want it:

- Strong match for `document/page/patent/real-world` weak domains.
- Very large, so we should sample instead of ingesting everything.

Where to store:

```text
V2-1/data/public_sources/patcid/
```

Recommended landing layout:

```text
V2-1/data/public_sources/patcid/
  raw/
  sampled/
  manifests/
  README_source.md
```

What to extract:

- image crops or source pointers
- structure labels
- provenance / license notes

How to use:

- Do not try to download/process the full corpus first.
- Start with a sampled subset focused on:
  - document context
  - embedded structures
  - scanned/patent-like visuals
- Convert only molecule crops with reliable labels into our weak-domain pool.

Recommendation:

- Training: yes, sampled subset
- Evaluation: only if provenance and label quality are very clean

## 3. PubChem

Official entry points:

- Download overview: [PubChem Downloads](https://pubchem.ncbi.nlm.nih.gov/docs/downloads)
- FTP root: [PubChem FTP](https://ftp.ncbi.nlm.nih.gov/pubchem/)

Why we want it:

- Best large-scale public source of molecules for self-rendered evaluation/training generation.
- We use it mainly as a SMILES source, not as a ready-made OCSR image dataset.

Where to store:

```text
V2-1/data/public_sources/pubchem/
```

Recommended landing layout:

```text
V2-1/data/public_sources/pubchem/
  raw/
  sampled_smiles/
  manifests/
  README_source.md
```

What to download:

- Compound tables containing canonical SMILES or SDF-derived structures.

Best use in this project:

- Sample diverse molecules.
- Render them ourselves into:
  - clean structure images
  - exam pages
  - printed-page layouts
  - photo/scan simulation pages

Recommendation:

- Training: yes
- Evaluation: yes, especially for self-generated controlled eval sets

## 4. ChEMBL

Official entry points:

- Main site: [ChEMBL](https://www.ebi.ac.uk/chembl)
- Access/download guide: [Accessing ChEMBL data](https://www.ebi.ac.uk/training/online/courses/chembl-quick-tour/accessing-chembl-data/)
- FTP root: [ChEMBL FTP](https://ftp.ebi.ac.uk/pub/databases/chembl/)

Why we want it:

- Good medicinal-chemistry molecule distribution.
- Strong complement to PubChem for realistic drug-like structures and stereo-rich molecules.

Where to store:

```text
V2-1/data/public_sources/chembl/
```

Recommended landing layout:

```text
V2-1/data/public_sources/chembl/
  raw/
  sampled_smiles/
  manifests/
  README_source.md
```

Best use in this project:

- Sample compounds with:
  - heterocycles
  - stereochemistry
  - longer drug-like structures
- Feed them into our self-generated evaluation and weak-domain training pipelines.

Recommendation:

- Training: yes
- Evaluation: yes, especially long/stereo stress subsets

## Download Priority

If we want the fastest value:

1. `DECIMER`
2. `PubChem`
3. `ChEMBL`
4. `PatCID`

Reason:

- `DECIMER` most directly fixes a weak domain.
- `PubChem/ChEMBL` are easiest to turn into controllable generated eval/training assets.
- `PatCID` is powerful but heavier to curate.

## What To Actually Do Next

### Fastest path

1. Download DECIMER into:

```text
V2-1/data/public_sources/decimer_handdrawn/raw/
```

2. Download or sample PubChem/ChEMBL SMILES into:

```text
V2-1/data/public_sources/pubchem/sampled_smiles/
V2-1/data/public_sources/chembl/sampled_smiles/
```

3. Use the self-generated evaluation pipeline in this workspace to turn sampled SMILES into:

- printed-page style images
- exam-page style images
- photo-simulated images
- scan-simulated images

4. If needed later, add a sampled PatCID subset for document realism.

### Scripts Added In This Workspace

Direct download helper:

```text
V2-1/scripts/download_public_weak_sources.ps1
```

Example:

```powershell
powershell -ExecutionPolicy Bypass -File V2-1/scripts/download_public_weak_sources.ps1 -Sources decimer
```

Download DECIMER and PatCID together:

```powershell
powershell -ExecutionPolicy Bypass -File V2-1/scripts/download_public_weak_sources.ps1 -Sources decimer,patcid -IncludeLarge
```

Sample PubChem/ChEMBL text tables into evaluation-seed SMILES:

```text
V2-1/scripts/sample_public_smiles_seed.py
```

Example:

```bash
python V2-1/scripts/sample_public_smiles_seed.py \
  --input V2-1/data/public_sources/pubchem/raw/pubchem_smiles.txt \
  --output V2-1/data/eval_generated/source_smiles/pubchem_seed.csv \
  --source-name pubchem \
  --limit 200
```

Then generate controlled evaluation images:

```bash
python V2-1/scripts/generate_controlled_eval_from_smiles.py \
  --input V2-1/data/eval_generated/source_smiles/pubchem_seed.csv \
  --output-root V2-1/data/eval_generated/generated_eval_v1
```

## Notes On Competition Reporting

- Public training sources can be listed in the report without shipping all raw assets.
- Private training data can be described without releasing it.
- Evaluation sets used for competition materials should clearly separate:
  - public source derived
  - self-generated controlled
  - private self-collected

Keep provenance files with every source under `README_source.md`.

# Data and evaluator provenance

## Original H1 inputs

Setup downloads the following objects directly from Arc's public
`arc-institute-virtual-cell-atlas` Google Cloud Storage bucket. They are not
included in this repository or its releases.

| Object | Generation | Bytes | SHA-256 |
|---|---:|---:|---|
| `2025/train/adata_Training.h5ad` | `1765904883947296` | 15,482,497,461 | `a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b` |
| `2025/train/pert_counts_Training.csv` | `1765906581937013` | 2,824 | `633d202be221418bdbac16efd8cb169666de0ef6469a4e9bfdfb64666ea81a89` |
| `2025/gene_names.csv` | `1765906114880540` | 116,023 | `e29ff3512b4c2440759bfb2a34577049b2f49df786b3e19c2a91b565de47b38e` |

The H1 object contains 221,273 cells by 18,080 genes with raw UMI counts in
CSR `X`. The benchmark deterministically samples 400 cells from each of the 126
targets having at least 400 cells using `default_rng(crc32(target))`, while
preserving target order from `pert_counts_Training.csv`.

## Derived release asset

`vcc2026-h1-benchmark-v1.zip` contains no single-cell count matrix. It contains
the fixed reference-cell list, reference DE and rank tables, pseudobulk moments,
the generic-response baseline, the five-split replicate anchor, and checksummed
manifests. Its SHA-256 is
`69d85faff91516d558c553711f494176d90b49b301ebd3a89695ca78f87a12fa`.

The control-only count object is reconstructed locally from Arc's source. It
contains all 38,176 cells labelled `non-targeting`, the complete H1 gene axis,
and the original `target_gene`, `guide_id`, and `batch` metadata.

## Evaluator

- `cell-eval2==0.16.0`
- source commit `5e64833518a6603a0301cbe28185d49c30f4a986`
- wheel SHA-256 `c78428ba705a94536e4a55464a34d1905aa5730d4f7e52ea8dbef4e7171d4fbe`
- `pdex==0.3.0`
- wheel SHA-256 `fa7d805925c5ae16e3b060390bc47c47b7f79cd162cbd6a86544963ab6a44739`
- resolved configuration SHA-256 `e0893621789230be31d942139353568bf454bd5f168b68b5ee5ca411779d4591`

Code in this repository is MIT licensed. The derived benchmark artifacts are
provided with their source provenance and do not relicense Arc's underlying H1
dataset.

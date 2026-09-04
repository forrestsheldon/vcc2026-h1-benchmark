import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from vcc_h1_eval.controls import build, sha256


def test_builds_exact_control_only_h5ad(tmp_path):
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 2, 0],
                [0, 3, 0, 4],
                [5, 0, 6, 0],
                [0, 7, 0, 8],
            ],
            dtype=np.float32,
        )
    )
    obs = pd.DataFrame(
        {
            "target_gene": ["non-targeting", "GENE1", "non-targeting", "GENE2"],
            "guide_id": ["nt1", "g1", "nt2", "g2"],
            "batch": ["a", "a", "b", "b"],
        },
        index=["cell0", "cell1", "cell2", "cell3"],
    )
    var = pd.DataFrame(
        {"gene_id": ["id0", "id1", "id2", "id3"]}, index=["A", "B", "C", "D"]
    )
    source = tmp_path / "source.h5ad"
    output = tmp_path / "controls.h5ad"
    manifest = tmp_path / "controls.json"
    ad.AnnData(matrix, obs=obs, var=var).write_h5ad(source)

    build(source, output, manifest, block_rows=1)

    result = ad.read_h5ad(output)
    assert result.X.dtype == np.uint32
    assert result.obs_names.tolist() == ["cell0", "cell2"]
    assert result.var_names.tolist() == ["A", "B", "C", "D"]
    assert result.obs.astype(str).equals(obs.iloc[[0, 2]].astype(str))
    assert np.array_equal(result.X.toarray(), matrix[[0, 2]].toarray())
    metadata = json.loads(manifest.read_text())
    assert metadata["artifact"]["shape"] == [2, 4]
    assert metadata["artifact"]["nnz"] == 4
    assert metadata["artifact"]["sha256"] == sha256(output)

from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_path


@dataclass(frozen=True)
class BenchmarkPaths:
    root: Path

    @classmethod
    def resolve(cls, value: Path | None = None) -> "BenchmarkPaths":
        root = value or user_cache_path("vcc2026-h1-benchmark")
        return cls(Path(root).expanduser().resolve())

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def h1(self) -> Path:
        return self.source_dir / "adata_Training.h5ad"

    @property
    def genes(self) -> Path:
        return self.source_dir / "gene_names.csv"

    @property
    def target_counts(self) -> Path:
        return self.source_dir / "pert_counts_Training.csv"

    @property
    def controls(self) -> Path:
        return self.root / "h1_controls.h5ad"

    @property
    def controls_manifest(self) -> Path:
        return self.root / "h1_controls_manifest.json"

    @property
    def benchmark_dir(self) -> Path:
        return self.root / "benchmark"

    @property
    def benchmark_manifest(self) -> Path:
        return self.benchmark_dir / "benchmark_manifest.json"

    @property
    def reference_cells(self) -> Path:
        return self.benchmark_dir / "reference_cells.csv"

    @property
    def reference_cache(self) -> Path:
        return self.benchmark_dir / "reference_cache"

    @property
    def scale(self) -> Path:
        return self.benchmark_dir / "scale"

    @property
    def score_cache(self) -> Path:
        return self.root / "score_cache"

    @property
    def installation_manifest(self) -> Path:
        return self.root / "installation.json"

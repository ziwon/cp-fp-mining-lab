from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .detector import FpDetector


STAGES = ("candidate", "staging", "production")


class LocalModelRegistry:
    """Filesystem model registry — an offline stand-in for the W&B Registry.

    Layout::

        <root>/models/<version>/model.joblib
        <root>/models/<version>/meta.json
        <root>/aliases.json            # {"production": <version>, ...}

    Mirrors the lineage concepts used in production (versions, stages, aliases,
    a promotion gate) without requiring a W&B account.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.aliases_path = self.root / "aliases.json"

    # --- aliases ---------------------------------------------------------
    def _read_aliases(self) -> dict[str, str]:
        if self.aliases_path.exists():
            return json.loads(self.aliases_path.read_text(encoding="utf-8"))
        return {}

    def _write_aliases(self, aliases: dict[str, str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.aliases_path.write_text(json.dumps(aliases, indent=2), encoding="utf-8")

    def stage_version(self, stage: str) -> str | None:
        return self._read_aliases().get(stage)

    def production_version(self) -> str | None:
        return self.stage_version("production")

    # --- versions --------------------------------------------------------
    def register(self, detector: FpDetector, stage: str = "candidate") -> str:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}; choose from {STAGES}.")
        version = detector.model_version
        vdir = self.models_dir / version
        detector.save(vdir / "model.joblib")
        meta = {
            "version": version,
            "stage": stage,
            "metrics": detector.metrics,
            "classes": detector.classes,
            "trained_at": detector.trained_at,
        }
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return version

    def register_file(
        self,
        version: str,
        weights_path: str | Path,
        metrics: dict[str, Any],
        stage: str = "candidate",
        filename: str = "model.pt",
        extra_meta: dict[str, Any] | None = None,
    ) -> str:
        """Register an arbitrary model weight file (e.g. a YOLO ``.pt``).

        Parallel to ``register`` (which stores a joblib ``FpDetector``); used for
        the detection track where the artifact is a checkpoint file, not a
        pickled estimator.
        """
        import shutil

        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}; choose from {STAGES}.")
        vdir = self.models_dir / version
        vdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(weights_path), str(vdir / filename))
        meta = {
            "version": version,
            "stage": stage,
            "metrics": metrics,
            "weights_file": filename,
            **(extra_meta or {}),
        }
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return version

    def weights_path(self, alias_or_version: str = "production") -> Path | None:
        """Path to a registered weight file by alias or version, or None."""
        version = self.stage_version(alias_or_version) or alias_or_version
        meta_path = self.models_dir / version / "meta.json"
        if not meta_path.exists():
            return None
        filename = json.loads(meta_path.read_text(encoding="utf-8")).get("weights_file", "model.pt")
        path = self.models_dir / version / filename
        return path if path.exists() else None

    def meta(self, version: str) -> dict[str, Any]:
        return json.loads((self.models_dir / version / "meta.json").read_text(encoding="utf-8"))

    def load(self, alias_or_version: str = "production") -> FpDetector:
        version = self.stage_version(alias_or_version) or alias_or_version
        path = self.models_dir / version / "model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No model for alias/version {alias_or_version!r} ({version}).")
        return FpDetector.load(path)

    def promote(self, version: str, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}; choose from {STAGES}.")
        if not (self.models_dir / version / "meta.json").exists():
            raise FileNotFoundError(f"Cannot promote unknown version {version!r}.")
        aliases = self._read_aliases()
        aliases[stage] = version
        self._write_aliases(aliases)
        meta = self.meta(version)
        meta["stage"] = stage
        (self.models_dir / version / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def maybe_promote(
        self,
        detector: FpDetector,
        metric_key: str = "macro_f1",
        min_delta: float = 0.0,
    ) -> dict[str, Any]:
        """Register ``detector`` as a candidate and promote it if it clears the gate.

        Promotes to ``production`` when there is no current production model, or
        when the candidate's metric beats production's by at least ``min_delta``.
        """
        version = self.register(detector, stage="candidate")
        candidate_metric = float(detector.metrics.get(metric_key, 0.0))

        prod = self.production_version()
        prod_metric = float(self.meta(prod)["metrics"].get(metric_key, 0.0)) if prod else None

        promoted = prod is None or candidate_metric >= prod_metric + min_delta
        if promoted:
            self.promote(version, "staging")
            self.promote(version, "production")
            reason = (
                "first model"
                if prod is None
                else f"{metric_key} {candidate_metric:.4f} >= {prod_metric:.4f} + {min_delta}"
            )
        else:
            reason = (
                f"{metric_key} {candidate_metric:.4f} < {prod_metric:.4f} + {min_delta}"
            )

        return {
            "version": version,
            "promoted": promoted,
            "reason": reason,
            "candidate_metric": candidate_metric,
            "previous_production_metric": prod_metric,
            "metric_key": metric_key,
        }

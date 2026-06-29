from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|Si|Se|Na|Li|Mg|Ca|Fe|Zn|Cu|Mn|Hg|Ag|Au|Al|As|"
    r"B|C|N|O|P|S|F|I|b|c|n|o|p|s|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|"
    r"\*|\$|\%[0-9]{2}|[0-9])"
)

ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()=.#+-\\/:@*%$")

FEATURE_NAMES = [
    "valid",
    "raw_len",
    "canonical_len",
    "raw_len_sq",
    "token_count",
    "token_mismatch",
    "unique_char_count",
    "digit_count",
    "ring_digit_count",
    "ring_digit_imbalance",
    "branch_count",
    "bond_count",
    "bracket_count",
    "aromatic_count",
    "upper_count",
    "lower_count",
    "stereo_count",
    "dot_count",
    "plus_count",
    "hyphen_count",
    "invalid_char_count",
    "heavy_atoms",
    "bond_count_mol",
    "ring_count",
    "chiral_centers",
    "hetero_atoms",
    "prompt_index",
    "tta_index",
    "vote_count",
]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def canonicalize_smiles(smiles_text: str) -> str | None:
    text = re.sub(r"\s+", "", normalize_text(smiles_text))
    if not text or Chem is None:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def extract_candidate_text(candidate: dict[str, Any]) -> str:
    for key in ("prediction", "raw_text", "canonical_prediction", "text"):
        value = candidate.get(key)
        if value:
            return normalize_text(value)
    return ""


def _ring_digit_imbalance(text: str) -> int:
    digits = [ch for ch in text if ch.isdigit()]
    if not digits:
        return 0
    counts = Counter(digits)
    return sum(count % 2 for count in counts.values())


def _mol_features(smiles: str) -> tuple[float, float, float, float, float]:
    if Chem is None or not smiles:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    heavy_atoms = float(mol.GetNumHeavyAtoms())
    bond_count = float(mol.GetNumBonds())
    ring_count = float(mol.GetRingInfo().NumRings())
    chiral_centers = float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)))
    hetero_atoms = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)))
    return heavy_atoms, bond_count, ring_count, chiral_centers, hetero_atoms


def feature_dict(candidate: dict[str, Any]) -> dict[str, float]:
    raw_text = re.sub(r"\s+", "", extract_candidate_text(candidate))
    canonical_text = candidate.get("canonical_prediction") or canonicalize_smiles(raw_text) or ""
    valid = 1.0 if canonical_text else 0.0
    token_list = SMILES_TOKEN_PATTERN.findall(raw_text)
    token_join = "".join(token_list)
    token_count = float(len(token_list) if token_list else len(raw_text))
    token_mismatch = 0.0 if not raw_text else (0.0 if token_join == raw_text else 1.0)
    digit_count = float(sum(ch.isdigit() for ch in raw_text))
    ring_digits = [ch for ch in raw_text if ch.isdigit()]
    ring_digit_count = float(len(set(ring_digits)))
    branch_count = float(raw_text.count("(") + raw_text.count(")"))
    bond_count = float(raw_text.count("=") + raw_text.count("#") + raw_text.count("/") + raw_text.count("\\"))
    bracket_count = float(raw_text.count("[") + raw_text.count("]"))
    aromatic_count = float(sum(ch in "bcnops" for ch in raw_text))
    upper_count = float(sum(ch.isupper() for ch in raw_text))
    lower_count = float(sum(ch.islower() for ch in raw_text))
    stereo_count = float(raw_text.count("@"))
    dot_count = float(raw_text.count("."))
    plus_count = float(raw_text.count("+"))
    hyphen_count = float(raw_text.count("-"))
    invalid_char_count = float(sum(ch not in ALLOWED_CHARS for ch in raw_text))
    heavy_atoms, bond_count_mol, ring_count, chiral_centers, hetero_atoms = _mol_features(canonical_text)
    return {
        "valid": valid,
        "raw_len": float(len(raw_text)),
        "canonical_len": float(len(canonical_text)),
        "raw_len_sq": float(len(raw_text) ** 2),
        "token_count": token_count,
        "token_mismatch": token_mismatch,
        "unique_char_count": float(len(set(raw_text))),
        "digit_count": digit_count,
        "ring_digit_count": ring_digit_count,
        "ring_digit_imbalance": float(_ring_digit_imbalance(raw_text)),
        "branch_count": branch_count,
        "bond_count": bond_count,
        "bracket_count": bracket_count,
        "aromatic_count": aromatic_count,
        "upper_count": upper_count,
        "lower_count": lower_count,
        "stereo_count": stereo_count,
        "dot_count": dot_count,
        "plus_count": plus_count,
        "hyphen_count": hyphen_count,
        "invalid_char_count": invalid_char_count,
        "heavy_atoms": heavy_atoms,
        "bond_count_mol": bond_count_mol,
        "ring_count": ring_count,
        "chiral_centers": chiral_centers,
        "hetero_atoms": hetero_atoms,
        "prompt_index": float(candidate.get("prompt_index", 0)),
        "tta_index": float(candidate.get("tta_index", 0)),
        "vote_count": float(candidate.get("vote_count", 0)),
    }


def feature_vector(candidate: dict[str, Any], feature_names: list[str] | None = None) -> np.ndarray:
    names = feature_names or FEATURE_NAMES
    feats = feature_dict(candidate)
    return np.asarray([feats.get(name, 0.0) for name in names], dtype=np.float32)


def make_policy(weights: np.ndarray, mean: np.ndarray, std: np.ndarray, bias: float = 0.0) -> dict[str, Any]:
    return {
        "feature_names": list(FEATURE_NAMES),
        "weights": weights.astype(float).tolist(),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "bias": float(bias),
    }


def load_policy(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_policy(path: str | Path, policy: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score_candidates(candidates: list[dict[str, Any]], policy: dict[str, Any]) -> np.ndarray:
    feature_names = list(policy.get("feature_names") or FEATURE_NAMES)
    weights = np.asarray(policy["weights"], dtype=np.float32)
    mean = np.asarray(policy.get("mean", [0.0] * len(feature_names)), dtype=np.float32)
    std = np.asarray(policy.get("std", [1.0] * len(feature_names)), dtype=np.float32)
    std = np.where(std == 0, 1.0, std)
    matrix = np.stack([feature_vector(candidate, feature_names) for candidate in candidates], axis=0)
    normalized = (matrix - mean) / std
    return normalized @ weights + float(policy.get("bias", 0.0))


def rank_candidates(candidates: list[dict[str, Any]], policy: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if policy is None:
        return sorted(
            candidates,
            key=lambda item: (
                1 if item.get("canonical_prediction") else 0,
                float(item.get("vote_count", 0)),
                -float(item.get("prompt_index", 0)),
                -float(item.get("tta_index", 0)),
                -len(str(item.get("prediction", ""))),
            ),
            reverse=True,
        )
    scores = score_candidates(candidates, policy)
    ranked = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["policy_score"] = float(score)
        ranked.append(item)
    ranked.sort(
        key=lambda item: (
            item["policy_score"],
            1 if item.get("canonical_prediction") else 0,
            float(item.get("vote_count", 0)),
            -float(item.get("prompt_index", 0)),
            -float(item.get("tta_index", 0)),
            -len(str(item.get("prediction", ""))),
        ),
        reverse=True,
    )
    return ranked

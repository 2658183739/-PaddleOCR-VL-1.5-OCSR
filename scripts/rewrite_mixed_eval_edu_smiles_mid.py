from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from rdkit import Chem

EXPLICIT_VAR_TOKENS = [" R ", " R^", " R _", "R ^", "R_", " X ", " Y ", " Z ", " R'", "\\prime"]
REACTION_TOKENS = ["xrightarrow", "\\rightarrow", "->", "+ ", " +", "xrightleftharpoons"]

bond_types = ["-", "=", "~", ">", "<", ">:", "<:", ">|", "<|", "-:", "=_", "=^", "~/"]
bond_types = sorted(bond_types, key=lambda x: -len(x))
pair_dict = {"}": "{", "]": "["}


class Atom:
    index = 0

    def __init__(self, text=""):
        self.name = f"Atom_{Atom.index}"
        Atom.index += 1
        self.m_text = text
        self.in_bonds = []
        self.out_bonds = []


class Bond:
    index = 0

    def __init__(self, b_type="-"):
        self.name = f"Bond_{Bond.index}"
        Bond.index += 1
        self.m_type = b_type.replace("_", "").replace("^", "")
        self.m_angle = None
        self.m_length = 1.0
        self.begin_atom = None
        self.end_atom = None


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "source", "image", "task_type", "image_type", "difficulty", "label_summary", "eval_target", "benchmark_track", "qc_status"])
        for row in records:
            writer.writerow([
                row["id"],
                row["source"],
                row["image"],
                row["task_type"],
                row["image_type"],
                row["difficulty"],
                row.get("label_summary", ""),
                row["eval_target"],
                row["benchmark_track"],
                row.get("qc_status", "pass"),
            ])


def copy_if_needed(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)


def is_middle_candidate(text: str) -> bool:
    if any(tok in text for tok in EXPLICIT_VAR_TOKENS):
        return False
    if any(tok in text for tok in REACTION_TOKENS):
        return False
    if "\\circle" in text:
        return False
    if "?[" in text:
        return False
    return True


def replace_chemfig(text: str):
    replace_dict = {}
    ind = 0
    while True:
        pos = text.find("\\chemfig")
        if pos == -1:
            break
        cur_pos = pos + 8
        cur_left_pair = None
        cur_left_pos = None
        cur_level = 0
        range_cnt = {"[": 0, "{": 0}
        while cur_pos < len(text):
            ch = text[cur_pos]
            if ch in "[{":
                if cur_left_pair is None:
                    cur_left_pair = ch
                    cur_left_pos = cur_pos
                    cur_level = 1
                elif cur_left_pair == ch:
                    cur_level += 1
            elif ch in "}]":
                if cur_left_pair == pair_dict[ch]:
                    cur_level -= 1
                    if cur_level == 0:
                        range_cnt[cur_left_pair] += 1
                        if range_cnt["{"] >= 1:
                            break
                        cur_left_pair = None
                        cur_left_pos = None
            elif cur_left_pair is None and ch != " ":
                break
            cur_pos += 1
        if cur_left_pos is None:
            text = text[pos + 8 :]
            continue
        begin_pos = cur_left_pos
        end_pos = cur_pos + 1
        rep_key = f"\\chem{chr(ord('a') + ind)}"
        ind += 1
        replace_dict[rep_key] = "\\chemfig " + text[begin_pos:end_pos]
        text = text[:pos] + " " + rep_key + " " + text[end_pos:]
    return replace_dict


def judge_str_item_type(item: str):
    if "?" in item:
        begin_result = re.compile(r"\?\[[a-zA-Z]+\]").findall(item)
        return "reconn_begin" if begin_result else "reconn_end"
    for bond in bond_types:
        if item.startswith(bond + "["):
            return "bond_atom"
    atom_result = re.compile(r"[a-zA-Z]+|\\circle").findall(item)
    if atom_result and atom_result[0] == item:
        return "atom"
    if item == "branch(":
        return "branch_begin"
    if item == "branch)":
        return "branch_end"
    return "atom"


def get_atom_group(item_list):
    out = []
    start = None
    for i, item in enumerate(item_list):
        item_type = judge_str_item_type(item)
        if item_type == "atom":
            if start is None:
                start = i
                out.append(item)
            else:
                out[-1] += item
        else:
            if start is not None:
                start = None
            out.append(item)
    return out


def attr_obtain(s: str):
    item_type = judge_str_item_type(s)
    if item_type == "bond_atom":
        bond_type = re.compile(r".*\[").findall(s)[0][:-1]
        attr_list = re.compile(r"[\d\.]+").findall(s)
        return bond_type, float(attr_list[0]), float(attr_list[1]) if len(attr_list) >= 2 else 1.0
    if item_type == "reconn_end":
        return re.compile(r"\{(.+)\}").findall(s)[0]
    return s


def add_atom(atom_dict, node_tag, name=""):
    if node_tag not in atom_dict:
        atom_dict[node_tag] = Atom(name)
    elif atom_dict[node_tag].m_text == "":
        atom_dict[node_tag].m_text = name


def parse_ssml(inner: str):
    Atom.index = 0
    Bond.index = 0
    item_list = get_atom_group(inner.split())
    branch_stack = [None for _ in range(1000)]
    stack_level = 0
    is_branch_end = False
    node_tag = 0
    atom_dict = {}
    reconn_begin_atom_dict = {}
    target_stack_level = 0

    for ssml_item in item_list:
        item_type = judge_str_item_type(ssml_item)
        if item_type == "atom":
            add_atom(atom_dict, node_tag, ssml_item)
        elif item_type == "bond_atom":
            add_atom(atom_dict, node_tag + 1)
            bond_type, bond_angle, bond_length = attr_obtain(ssml_item)
            bond = Bond(bond_type)
            bond.m_angle = bond_angle
            bond.m_length = bond_length
            bond.end_atom = atom_dict[node_tag + 1]
            atom_dict[node_tag + 1].in_bonds.append(bond)
            if is_branch_end:
                branch_begin_tag = branch_stack[target_stack_level]
                is_branch_end = False
                bond.begin_atom = atom_dict[branch_begin_tag]
                atom_dict[branch_begin_tag].out_bonds.append(bond)
            else:
                add_atom(atom_dict, node_tag)
                bond.begin_atom = atom_dict[node_tag]
                atom_dict[node_tag].out_bonds.append(bond)
            node_tag += 1
        elif item_type == "reconn_begin":
            tag = re.match(r"\?\[([a-zA-Z]+)\]", ssml_item).group(1)
            reconn_begin_atom_dict[tag] = node_tag
        elif item_type == "reconn_end":
            cur_reconn_tag, bond_type = re.match(r"\?\[([a-zA-Z]+)[,]\{([^\{\}]+)\}\]", ssml_item).groups()
            reconn_atom = reconn_begin_atom_dict[cur_reconn_tag]
            bond = Bond(bond_type)
            bond.begin_atom = atom_dict[reconn_atom]
            bond.end_atom = atom_dict[node_tag]
            atom_dict[reconn_atom].out_bonds.append(bond)
            atom_dict[node_tag].in_bonds.append(bond)
        elif item_type == "branch_begin":
            if not is_branch_end:
                branch_stack[stack_level] = node_tag
            stack_level += 1
        elif item_type == "branch_end":
            stack_level -= 1
            target_stack_level = stack_level
            is_branch_end = True
    return atom_dict


SINGLE_ATOM_SYMBOLS = {
    "": ("C", False),
    "C": ("C", False),
    "CH": ("C", False),
    "CH_{2}": ("C", False),
    "CH_{3}": ("C", False),
    "H_{3}C": ("C", False),
    "H_{2}C": ("C", False),
    "O": ("O", False),
    "OH": ("O", False),
    "HO": ("O", False),
    "N": ("N", False),
    "NH": ("N", False),
    "H_{2}N": ("N", False),
    "Cl": ("Cl", False),
    "Br": ("Br", False),
    "F": ("F", False),
    "S": ("S", False),
    "Al": ("Al", False),
    "Al.": ("Al", False),
    "H": ("H", True),
}

FRAGMENT_SPECS = {
    "CH_{2}OH": ("CO", 0),
    "CH_{2}CH_{2}OH": ("CCO", 0),
    "CH_{2}Br": ("CBr", 0),
    "CH_{2}Cl": ("CCl", 0),
    "CHBr": ("CBr", 0),
    "CHO": ("C=O", 0),
    "OHC": ("C=O", 0),
    "COOH": ("C(=O)O", 0),
    "HCOOH": ("C(=O)O", 0),
    "COOC_{2}H_{5}": ("C(=O)OCC", 0),
    "COOCH_{3}": ("C(=O)OC", 0),
    "COO": ("C(=O)O", 0),
    "C_{2}H_{5}": ("CC", 0),
    "CC": ("CC", 0),
    "CH_{3}CH_{2}": ("CC", 1),
    "CH_{3}CH": ("CC", 1),
    "CHCH_{3}": ("CC", 0),
    "CH_{3}C": ("CC", 1),
    "CCH_{3}": ("CC", 0),
    "CH_{2}CHO": ("CC=O", 0),
    "CHOH": ("CO", 0),
    "CN": ("CN", 0),
    "COONa": ("C(=O)[O-].[Na+]", 0),
    "FC": ("CF", 1),
    "C(CH_{3})_{2}": ("C(C)C", 0),
    "(CH_{3})_{2}C": ("C(C)C", 0),
    "OCH_{2}CH_{2}OH": ("OCCO", 0),
    "HOH_{2}C": ("OC", 1),
}

UNSUPPORTED_TOKENS = {
    ".",
    "CH_{5}",
    "OH_{3}CH_{2}",
    "C_{2}H_{3}O",
    "OCH_{2}",
}

BOND_TYPE_MAP = {
    "-": Chem.rdchem.BondType.SINGLE,
    "=": Chem.rdchem.BondType.DOUBLE,
    "~": Chem.rdchem.BondType.TRIPLE,
    ">": Chem.rdchem.BondType.SINGLE,
    "<": Chem.rdchem.BondType.SINGLE,
    ">:": Chem.rdchem.BondType.SINGLE,
    "<:": Chem.rdchem.BondType.SINGLE,
    ">|": Chem.rdchem.BondType.SINGLE,
    "<|": Chem.rdchem.BondType.SINGLE,
    "-:": Chem.rdchem.BondType.SINGLE,
    "=_": Chem.rdchem.BondType.DOUBLE,
    "=^": Chem.rdchem.BondType.DOUBLE,
    "~/": Chem.rdchem.BondType.TRIPLE,
}


def normalize_token(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def append_fragment(mol: Chem.RWMol, fragment: Chem.Mol):
    offset = mol.GetNumAtoms()
    for atom in fragment.GetAtoms():
        copied = Chem.Atom(atom.GetAtomicNum())
        copied.SetFormalCharge(atom.GetFormalCharge())
        copied.SetIsAromatic(atom.GetIsAromatic())
        copied.SetNoImplicit(atom.GetNoImplicit())
        copied.SetNumExplicitHs(atom.GetNumExplicitHs())
        mol.AddAtom(copied)
    for bond in fragment.GetBonds():
        mol.AddBond(offset + bond.GetBeginAtomIdx(), offset + bond.GetEndAtomIdx(), bond.GetBondType())
    return list(range(offset, offset + fragment.GetNumAtoms()))


def add_group(mol: Chem.RWMol, token: str):
    norm = normalize_token(token)
    if norm in SINGLE_ATOM_SYMBOLS:
        symbol, explicit_h = SINGLE_ATOM_SYMBOLS[norm]
        atom = Chem.Atom(symbol)
        if explicit_h:
            atom.SetNoImplicit(True)
        idx = mol.AddAtom(atom)
        return idx, None
    if norm in UNSUPPORTED_TOKENS:
        return -1, f"unsupported_token:{norm}"
    spec = FRAGMENT_SPECS.get(norm)
    if spec is None:
        return -1, f"unknown_token:{norm}"
    smiles, anchor = spec
    fragment = Chem.MolFromSmiles(smiles)
    if fragment is None:
        return -1, f"fragment_parse_failed:{norm}:{smiles}"
    indices = append_fragment(mol, fragment)
    return indices[anchor], None


def atoms_to_mol(atom_dict):
    mol = Chem.RWMol()
    anchor_map = {}
    for key in sorted(atom_dict.keys()):
        anchor_idx, error = add_group(mol, atom_dict[key].m_text)
        if error:
            return None, error
        anchor_map[key] = anchor_idx
    seen = set()
    reverse_lookup = {id(v): k for k, v in atom_dict.items()}
    for key in sorted(atom_dict.keys()):
        atom = atom_dict[key]
        for bond in atom.out_bonds:
            begin_key = reverse_lookup[id(bond.begin_atom)]
            end_key = reverse_lookup[id(bond.end_atom)]
            edge_key = tuple(sorted((begin_key, end_key)))
            if edge_key in seen:
                continue
            seen.add(edge_key)
            bond_type = BOND_TYPE_MAP.get(bond.m_type)
            if bond_type is None:
                return None, f"unsupported_bond_type:{bond.m_type}"
            try:
                mol.AddBond(anchor_map[begin_key], anchor_map[end_key], bond_type)
            except Exception as exc:
                return None, f"add_bond_failed:{begin_key}-{end_key}:{exc}"
    try:
        result = mol.GetMol()
        Chem.SanitizeMol(result)
        result = Chem.RemoveHs(result)
        Chem.SanitizeMol(result)
        return result, None
    except Exception as exc:
        return None, f"sanitize_failed:{exc}"


def ssml_to_smiles(ssml_normed: str):
    replace_dict = replace_chemfig(ssml_normed)
    if not replace_dict:
        return None, "no_chemfig"
    if len(replace_dict) != 1:
        return None, f"chemfig_count:{len(replace_dict)}"
    chemfig = next(iter(replace_dict.values()))
    inner = " ".join(chemfig.split(" ")[2:-1]).strip()
    atom_dict = parse_ssml(inner)
    if not atom_dict:
        return None, "parse_empty"
    mol, error = atoms_to_mol(atom_dict)
    if error:
        return None, error
    smiles = Chem.MolToSmiles(mol, canonical=True)
    return smiles, None


def select_middle_candidates(project_root: Path):
    labels_path = project_root / "V2" / "data" / "eval" / "edu_chemc_convertibility_trial_v1" / "annotations" / "labels.jsonl"
    rows = list(read_jsonl(labels_path))
    selected = []
    for row in rows:
        text = str(row.get("ssml_normed", ""))
        if is_middle_candidate(text):
            selected.append(row)
    return selected


def make_edu_row(row, smiles):
    return {
        "id": row["id"],
        "source": "edu_chemc",
        "image": f"images/edu_chemc/{Path(row['image']).name}",
        "task_type": "molecule_structure_recognition",
        "image_type": row.get("image_type", "handwritten_education_structure"),
        "difficulty": row.get("difficulty", "hard"),
        "ground_truth": {"smiles": smiles, "inchi": None, "selfies": None, "mol": None},
        "eval_target": "canonical_smiles",
        "license": row.get("license", "unknown_pending_confirmation"),
        "source_url_or_doc": row.get("source_url_or_doc", "local_EDU-CHMEC-MM23_bundle"),
        "qc_status": row.get("qc_status", "pass"),
        "benchmark_track": "edu_realworld",
        "label_summary": smiles,
    }


def backup_current_edu(current_root: Path, backup_root: Path):
    backup_root.mkdir(parents=True, exist_ok=True)
    annotations = current_root / "annotations"
    for name in ["labels.jsonl", "labels.csv"]:
        src = annotations / name
        if src.exists():
            copy_if_needed(src, backup_root / name)
    for name in ["stats.json", "subgroup_summary.json", "README.md", "TECHNICAL_REPORT_zh.md", "mixed_eval_summary.json"]:
        src = current_root / name
        if src.exists():
            copy_if_needed(src, backup_root / name)
    edu_dir = current_root / "images" / "edu_chemc"
    if edu_dir.exists():
        shutil.copytree(edu_dir, backup_root / "images_edu_chemc", dirs_exist_ok=True)


def build_dataset(project_root: Path, out_root: Path):
    current_root = project_root / "V2" / "data" / "eval" / "ocsr_realworld_mixed_eval_v1"
    trial_root = project_root / "V2" / "data" / "eval" / "edu_chemc_convertibility_trial_v1"
    backup_root = project_root / "V2" / "data" / "eval" / "_backups" / f"ocsr_realworld_mixed_eval_v1_before_mid_smiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tmp_root = current_root.parent / (current_root.name + "__tmp_mid_smiles")

    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    current_rows = list(read_jsonl(current_root / "annotations" / "labels.jsonl"))
    canonical_rows = [row for row in current_rows if row.get("source") != "edu_chemc"]

    for row in canonical_rows:
        src = current_root / row["image"]
        dest = tmp_root / row["image"]
        copy_if_needed(src, dest)

    middle_candidates = select_middle_candidates(project_root)
    converted_rows = []
    failures = []
    failure_counter = Counter()
    for row in middle_candidates:
        smiles, error = ssml_to_smiles(str(row.get("ssml_normed", "")))
        if error:
            failures.append({"id": row["id"], "image": row["image"], "error": error})
            failure_counter[error.split(":", 1)[0]] += 1
            continue
        converted_rows.append(make_edu_row(row, smiles))
        src = trial_root / row["image"]
        dest = tmp_root / f"images/edu_chemc/{Path(row['image']).name}"
        copy_if_needed(src, dest)

    mixed_rows = canonical_rows + sorted(converted_rows, key=lambda x: x["id"])
    write_jsonl(tmp_root / "annotations/labels.jsonl", mixed_rows)
    write_csv(tmp_root / "annotations/labels.csv", mixed_rows)

    by_track = Counter(r["benchmark_track"] for r in mixed_rows)
    by_source = Counter(r["source"] for r in mixed_rows)
    by_difficulty = Counter(r["difficulty"] for r in mixed_rows)

    stats = {
        "total": len(mixed_rows),
        "by_track": dict(by_track),
        "by_source": dict(by_source),
        "by_difficulty": dict(by_difficulty),
        "canonical_summary": {"before": 767, "removed": 150, "after": 617, "output_root": str(project_root / 'V2' / 'data' / 'eval' / 'canonical_smiles_curated_v2')},
        "edu_conversion_summary": {
            "source_trial_candidates": 460,
            "middle_candidates": len(middle_candidates),
            "converted_success": len(converted_rows),
            "converted_failures": len(failures),
            "middle_rule": "no_explicit_var_no_reaction_no_circle_no_reconnect",
            "backup_root": str(backup_root),
        },
    }
    (tmp_root / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    subgroup_summary = {
        "canonical_main": {
            "count": by_track.get("canonical_main", 0),
            "eval_target": "canonical_smiles",
            "sources": {k: v for k, v in by_source.items() if k in {"uob", "uspto", "real_world"}},
        },
        "edu_realworld": {
            "count": by_track.get("edu_realworld", 0),
            "eval_target": "canonical_smiles",
            "sources": {k: v for k, v in by_source.items() if k == "edu_chemc"},
            "selection": {
                "source_trial_candidates": 460,
                "middle_candidates": len(middle_candidates),
                "converted_success": len(converted_rows),
                "converted_failures": len(failures),
                "rule": "no_explicit_var_no_reaction_no_circle_no_reconnect",
            },
        },
        "overall": {"count": len(mixed_rows)},
    }
    (tmp_root / "subgroup_summary.json").write_text(json.dumps(subgroup_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = (
        "# OCSR 真实世界混合评测集 v1\n\n"
        "这是当前项目用于真实世界化学图像理解与跨域泛化评估的混合域评测集。\n\n"
        f"- canonical_main：{by_track.get('canonical_main', 0)}\n"
        f"- edu_realworld（canonical_smiles middle subset）：{by_track.get('edu_realworld', 0)}\n"
        f"- 总计：{len(mixed_rows)}\n\n"
        "当前版本已清理原先 EDU-CHEMC 的非 SMILES 输出，仅保留从中间候选集成功转换到 canonical SMILES 的子集。\n"
    )
    (tmp_root / "README.md").write_text(readme, encoding="utf-8")

    technical = (
        "# OCSR 真实世界混合评测集技术报告（中文）\n\n"
        "## 1. 数据集定位\n\n"
        "本版本保留 canonical_main 主任务子集，并将 EDU-CHEMC 部分重写为一个 canonical_smiles 的中间保守子集。\n\n"
        "## 2. 数据集规模\n\n"
        f"- 总样本数：{len(mixed_rows)}\n"
        f"- canonical_main：{by_track.get('canonical_main', 0)}\n"
        f"- edu_realworld：{by_track.get('edu_realworld', 0)}\n\n"
        "## 3. edu_realworld 重写说明\n\n"
        "- 来源：EDU-CHEMC test 子集\n"
        "- 入口候选：460\n"
        f"- 中间版候选：{len(middle_candidates)}\n"
        f"- 成功转换为 canonical SMILES：{len(converted_rows)}\n"
        f"- 转换失败：{len(failures)}\n"
        "- 筛选规则：无显式变量、无反应箭头、无 `\\circle`、无 `?[...]` 重连\n"
        "- 说明：已删除原先 edu_chemc 中非 SMILES 的输出\n\n"
        "## 4. 当前标签空间\n\n"
        "- canonical_main：canonical_smiles\n"
        "- edu_realworld：canonical_smiles\n\n"
        "## 5. 备注\n\n"
        f"详细失败原因见 `edu_conversion_mid_failures.json`，原目录关键文件的备份位于：`{backup_root}`。\n"
    )
    (tmp_root / "TECHNICAL_REPORT_zh.md").write_text(technical, encoding="utf-8")

    mixed_summary = {
        "canonical_total": by_track.get("canonical_main", 0),
        "edu_selected": by_track.get("edu_realworld", 0),
        "total": len(mixed_rows),
        "output_root": str(current_root),
    }
    (tmp_root / "mixed_eval_summary.json").write_text(json.dumps(mixed_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (tmp_root / "edu_conversion_mid_failures.json").write_text(json.dumps({"count": len(failures), "by_reason": dict(failure_counter), "rows": failures}, ensure_ascii=False, indent=2), encoding="utf-8")

    backup_current_edu(current_root, backup_root)
    if current_root.exists():
        shutil.rmtree(current_root)
    tmp_root.rename(current_root)

    return {
        "canonical_rows": len(canonical_rows),
        "middle_candidates": len(middle_candidates),
        "converted_success": len(converted_rows),
        "converted_failures": len(failures),
        "total": len(mixed_rows),
        "output_root": str(current_root),
        "backup_root": str(backup_root),
        "failure_reasons": dict(failure_counter),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    summary = build_dataset(project_root, project_root / "V2" / "data" / "eval" / "ocsr_realworld_mixed_eval_v1")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

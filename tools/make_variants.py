from copy import deepcopy
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[1]
full = yaml.safe_load((root / "configs/nert.yaml").read_text(encoding="utf-8"))
variants = {"baseline": (False, False, False), "dmo": (True, False, False),
            "c2ls_only": (False, True, False), "dmo_c2ls": (True, True, False)}
for name, (dmo, c2ls, nfeh) in variants.items():
    config = deepcopy(full)
    for layer in config["backbone"] + config["head"]:
        if layer[2] == "C3k2DMO" and not dmo:
            layer[2], layer[3] = "C3k2", layer[3][:5]
        if layer[2] == "C2LS" and not c2ls:
            layer[2], layer[3] = "C2PSA", layer[3][:1]
        if layer[2] == "NFEH" and not nfeh:
            layer[2] = "Detect"
    (root / "configs" / (name + ".yaml")).write_text(
        "# Architecture variant with fixed channel widths and repeat counts.\n" +
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print("Generated baseline, dmo, c2ls_only, dmo_c2ls (full model is nert.yaml).")

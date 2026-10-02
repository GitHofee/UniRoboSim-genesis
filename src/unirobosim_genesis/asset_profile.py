"""Explicit, hash-pinned USD asset selections for the official Genesis importer."""
from pathlib import Path
import hashlib
import json
from unirobosim import ValidationError


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AssetProfile:
    def __init__(self, filename=None):
        self.path = Path(filename).expanduser().resolve() if filename else None
        self.digest = sha256(self.path) if self.path else None
        self.entries = {}
        if self.path is None:
            return
        payload = json.loads(self.path.read_text())
        if payload.get("schema") != "unirobosim-genesis-asset-profile/v1":
            raise ValidationError("unsupported Genesis asset profile schema", operation="genesis.asset_profile")
        for entry in payload.get("entries", []):
            key = entry.get("source_sha256")
            if not isinstance(key, str) or len(key) != 64 or key in self.entries:
                raise ValidationError("invalid or duplicate source digest in asset profile", operation="genesis.asset_profile")
            self.entries[key] = entry

    def select(self, source):
        digest = sha256(source)
        entry = self.entries.get(digest)
        if entry is None:
            return None
        from pxr import Usd
        sources = {}
        for item in entry.get("source_files", []):
            path = self._path(item["file"])
            if sha256(path) != item["sha256"]:
                raise ValidationError("source USD dependency changed since asset repair", operation="genesis.asset_profile", details={"file": str(path)})
            sources[path] = item["sha256"]
        root = next((p for p, h in sources.items() if h == digest), None)
        if root is None:
            raise ValidationError("asset profile does not pin the original source", operation="genesis.asset_profile")
        stage = Usd.Stage.Open(str(root))
        if stage is None or any(not layer.anonymous and Path(layer.realPath).resolve() not in sources for layer in stage.GetUsedLayers()):
            raise ValidationError("source USD has an unpinned dependency", operation="genesis.asset_profile")
        pinned = {}
        for item in entry.get("pinned_files", []):
            path = self._path(item["file"])
            if sha256(path) != item["sha256"]:
                raise ValidationError("Genesis asset profile file digest mismatch", operation="genesis.asset_profile", details={"file": str(path)})
            pinned[path] = item["sha256"]
        loads = []
        for item in entry["loads"]:
            item = dict(item)
            path = self._path(item["file"])
            if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
                raise ValidationError("asset profiles load USD only", operation="genesis.asset_profile")
            if pinned.get(path) != item["sha256"]:
                raise ValidationError("USD is not pinned in asset profile", operation="genesis.asset_profile")
            if item.get("mode") not in {"add_entity", "add_stage"}:
                raise ValidationError("invalid USD load mode", operation="genesis.asset_profile")
            if {"file", "pos", "quat", "scale"} & item.get("morph_kwargs", {}).keys():
                raise ValidationError("profile cannot override entity placement", operation="genesis.asset_profile")
            stage = Usd.Stage.Open(str(path))
            if stage is None:
                raise ValidationError("profile USD cannot be opened", operation="genesis.asset_profile")
            for layer in stage.GetUsedLayers():
                if not layer.anonymous and Path(layer.realPath).resolve() not in pinned:
                    raise ValidationError("profile USD references an unpinned layer", operation="genesis.asset_profile", details={"layer": layer.realPath})
            item["file"] = str(path)
            loads.append(item)
        result = dict(entry, loads=loads, profile_sha256=self.digest)
        if "visual_usd" in entry:
            visual = dict(entry["visual_usd"])
            path = self._path(visual["file"])
            if path.suffix.lower() not in {".usd", ".usda", ".usdc"} or pinned.get(path) != visual["sha256"]:
                raise ValidationError("visual USD is not pinned in asset profile", operation="genesis.asset_profile")
            stage = Usd.Stage.Open(str(path))
            if stage is None or any(not layer.anonymous and Path(layer.realPath).resolve() not in pinned for layer in stage.GetUsedLayers()):
                raise ValidationError("visual USD has an unpinned layer", operation="genesis.asset_profile")
            result["visual_usd"] = dict(visual, file=str(path))
        if "robot_manifest" in entry:
            path = self._path(entry["robot_manifest"])
            if path not in pinned:
                raise ValidationError("robot manifest is not pinned", operation="genesis.asset_profile")
            result["robot_metadata"] = json.loads(path.read_text())
        return result

    def _path(self, value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.path.parent / path).resolve()

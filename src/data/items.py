"""아이템 목록을 Data Dragon에서 받아 둔다 (게임 후 피드백의 "첫 코어템" 판정). API 키가 필요 없는 공개 정적 데이터.

    python -m src.data.items

`data/items.json`에 {아이템ID: {이름, 가격, 완성템 여부}} 로 저장한다. 파일이 없으면 처음 쓸 때 받아 온다.
완성템 = 더 올라갈 데가 없고, 신발 · 소모품 · 장신구 · 서폿 퀘스트템이 아니고, 2,200골드 이상.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

from src.i18n import L

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "items.json"
CORE_MIN_GOLD = 2200
NOT_CORE_TAGS = {"Boots", "Consumable", "Trinket", "GoldPer"}


def fetch() -> dict:
    version = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=15).json()[0]
    data = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/item.json", timeout=15).json()["data"]
    items = {}
    for k, v in data.items():
        if not (v.get("maps") or {}).get("11"):          # 소환사의 협곡
            continue
        gold = v.get("gold") or {}
        core = (not v.get("into") and gold.get("purchasable", False) and gold.get("total", 0) >= CORE_MIN_GOLD
                and not (set(v.get("tags") or []) & NOT_CORE_TAGS))
        items[int(k)] = {"name": v["name"], "gold": gold.get("total", 0), "core": core}
    return {"version": version, "items": items}


def save() -> dict:
    r = fetch()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def load() -> dict[int, dict]:
    """파일이 없으면 한 번 받아 온다. 인터넷이 안 되면 빈 것 — 첫 코어템 항목만 빠진다."""
    if not OUT.exists():
        try:
            save()
        except (requests.RequestException, ValueError, KeyError, IndexError, OSError) as exc:
            print(L(f"아이템 목록을 받지 못했습니다: {exc}", f"Couldn't download the item list: {exc}"), file=sys.stderr)
            return {}
    try:
        return {int(k): v for k, v in json.loads(OUT.read_text(encoding="utf-8"))["items"].items()}
    except (OSError, ValueError, KeyError):
        return {}


if __name__ == "__main__":
    r = save()
    n, core = len(r["items"]), sum(v["core"] for v in r["items"].values())
    print(L(f"패치 {r['version']} · 아이템 {n}개 (완성템 {core}개) → {OUT}",
            f"Patch {r['version']} · {n} items ({core} completed) → {OUT}"))

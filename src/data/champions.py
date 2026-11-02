"""챔피언 목록을 Data Dragon에서 받아 둔다. API 키가 필요 없는 공개 정적 데이터.

    python -m src.data.champions

`data/champions.json`에 {챔피언ID: {한글명, 영문명, 이미지키, 초성}} 형태로 저장한다.
초성을 미리 만들어 두는 이유: 밴픽은 30초 안에 끝나서 "ㅇㅈ" 같은 검색이 필요하다.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from src.i18n import L

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "champions.json"

CHOSUNG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"


def chosung(text: str) -> str:
    """한글 문자열의 초성만 뽑는다. 한글이 아닌 글자는 그대로 둔다."""
    out = []
    for char in text:
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            out.append(CHOSUNG[(code - 0xAC00) // 588])
        elif char != " ":
            out.append(char)
    return "".join(out)


def fetch() -> dict:
    versions = requests.get(
        "https://ddragon.leagueoflegends.com/api/versions.json", timeout=15
    ).json()
    version = versions[0]
    data = requests.get(
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/ko_KR/champion.json",
        timeout=15,
    ).json()["data"]
    english = requests.get(
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json",
        timeout=15,
    ).json()["data"]

    en_by_key = {value["key"]: value["name"] for value in english.values()}
    champions = {}
    for value in data.values():
        champion_id = int(value["key"])
        champions[champion_id] = {
            "name": value["name"],
            "en": en_by_key.get(value["key"], value["id"]),
            "image": value["id"],  # 이미지 파일명에 쓰인다
            "chosung": chosung(value["name"]),
        }
    return {"version": version, "champions": champions}


def load() -> dict:
    if not OUT.exists():
        raise SystemExit(L("data/champions.json이 없습니다. `python -m src.data.champions` 먼저 실행하세요.",
                           "data/champions.json is missing. Run `python -m src.data.champions` first."))
    raw = json.loads(OUT.read_text(encoding="utf-8"))
    raw["champions"] = {int(k): v for k, v in raw["champions"].items()}
    return raw


if __name__ == "__main__":
    result = fetch()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(L(f"패치 {result['version']} · 챔피언 {len(result['champions'])}종 → {OUT}",
            f"Patch {result['version']} · {len(result['champions'])} champions → {OUT}"))

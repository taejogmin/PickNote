"""한국어 / English.

    L("한국어", "English")               # 0_Setup.cmd에서 고른 언어 (data/lang.txt)
    L("한국어", "English", lang="en")    # 화면이 보낸 언어로

문장을 표로 따로 빼지 않고 쓰는 자리에 둘 다 적는다 — 한 줄만 보고도 두 언어가 맞는지 알 수 있다.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANG_FILE = ROOT / "data" / "lang.txt"
LANGS = ("ko", "en")


def default_lang() -> str:
    try:
        lang = LANG_FILE.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return "ko"
    return lang if lang in LANGS else "ko"


def set_default_lang(lang: str) -> None:
    LANG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LANG_FILE.write_text(lang if lang in LANGS else "ko", encoding="utf-8")


def pick(lang: str | None) -> str:
    """화면이 보낸 값이 이상하면 기본 언어로."""
    return lang if lang in LANGS else default_lang()


def L(ko: str, en: str, lang: str | None = None) -> str:
    return en if pick(lang) == "en" else ko

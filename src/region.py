"""어느 서버의 데이터를 쓰나. 기본은 한국(kr).

    server()            0_Setup.cmd · Settings.cmd에서 고른 서버 (data/server.txt)
    platform(s)         리그 목록을 묻는 주소 — kr, na1, euw1 …
    region(s)           매치를 묻는 주소 — asia, americas, europe, sea
    db_path(s)          수집 DB. 한국은 data/matches.db, 그 밖은 data/<서버>/matches.db
    model_dir(s)        모델. 한국은 models/, 그 밖은 models/<서버>/

서버마다 DB와 모델을 따로 둔다 — 서버가 다르면 메타가 달라서 섞으면 둘 다 흐려진다.
들어 있는 모델은 한국 데이터로 학습한 것이라, 다른 서버는 직접 모은 경기로 학습하기 전까지 한국 모델을 빌려 쓴다.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_FILE = ROOT / "data" / "server.txt"
DEFAULT = "kr"

# 서버 코드: (매치 지역, 한국어 이름, 영어 이름)
SERVERS = {
    "kr":   ("asia",     "한국", "Korea"),
    "jp1":  ("asia",     "일본", "Japan"),
    "na1":  ("americas", "북미", "North America"),
    "br1":  ("americas", "브라질", "Brazil"),
    "la1":  ("americas", "라틴 아메리카 북부", "Latin America North"),
    "la2":  ("americas", "라틴 아메리카 남부", "Latin America South"),
    "euw1": ("europe",   "유럽 서부", "Europe West"),
    "eun1": ("europe",   "유럽 북동부", "Europe Nordic & East"),
    "tr1":  ("europe",   "튀르키예", "Türkiye"),
    "ru":   ("europe",   "러시아", "Russia"),
    "me1":  ("europe",   "중동", "Middle East"),
    "oc1":  ("sea",      "오세아니아", "Oceania"),
    "sg2":  ("sea",      "동남아시아", "Southeast Asia"),
    "tw2":  ("sea",      "대만", "Taiwan"),
    "vn2":  ("sea",      "베트남", "Vietnam"),
}


def server() -> str:
    try:
        value = SERVER_FILE.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return DEFAULT
    return value if value in SERVERS else DEFAULT


def set_server(value: str) -> None:
    SERVER_FILE.parent.mkdir(parents=True, exist_ok=True)
    SERVER_FILE.write_text(value if value in SERVERS else DEFAULT, encoding="utf-8")


def pick(s: str | None) -> str:
    s = (s or "").strip().lower()
    return s if s in SERVERS else server()


def platform(s: str | None = None) -> str:
    return pick(s)


def region(s: str | None = None) -> str:
    return SERVERS[pick(s)][0]


def account_region(s: str | None = None) -> str:
    """계정(Riot ID) 조회는 americas · asia · europe 셋뿐이다. 동남아 서버 계정은 asia로 묻는다."""
    r = region(s)
    return "asia" if r == "sea" else r


def region_of_match(match_id: str) -> str:
    """매치 ID 앞부분이 서버다 ("EUW1_1234" → europe). 모르는 꼴이면 지금 서버의 지역."""
    prefix = match_id.split("_", 1)[0].lower()
    return SERVERS[prefix][0] if prefix in SERVERS else region()


def name(s: str | None = None, lang: str | None = None) -> str:
    from src.i18n import L

    _r, ko, en = SERVERS[pick(s)]
    return L(ko, en, lang)


def data_dir(s: str | None = None) -> Path:
    s = pick(s)
    return ROOT / "data" if s == DEFAULT else ROOT / "data" / s


def db_path(s: str | None = None) -> Path:
    return data_dir(s) / "matches.db"


def model_dir(s: str | None = None) -> Path:
    s = pick(s)
    return ROOT / "models" if s == DEFAULT else ROOT / "models" / s


def usable_model_dir(s: str | None = None) -> Path:
    """그 서버 모델이 아직 없으면 들어 있는 한국 모델."""
    own = model_dir(s)
    return own if (own / "draft.npz").exists() else model_dir(DEFAULT)

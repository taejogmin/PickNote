# PickNote가 부르는 주소

PickNote가 어디에 무엇을 요청하는지 전부 적은 목록. Riot 개발자 포털에 제품을 등록할 때 이 내용을 그대로 쓴다.

## League Client API (LCU) — 내 PC 안

| 메서드 | 엔드포인트 | 언제 | 무엇을 쓰나 |
|---|---|---|---|
| GET | `/lol-champ-select/v1/session` | 화면이 켜져 있는 동안 1초마다 | 챔피언 선택 중인지(200 / 404), 양 팀이 고른·고민 중인(호버) 챔피언, 밴, 내 자리 |

- **읽기만 한다.** 픽 · 밴 · 채팅 등 상태를 바꾸는 요청(POST · PATCH · PUT · DELETE)은 코드에 없다 — `src/lcu/client.py`.
- 접속 정보(포트 · 비밀번호)는 클라이언트 설치 폴더의 `lockfile`에서, 없으면 `LeagueClientUx.exe` 실행 인자에서 읽는다. `127.0.0.1`로만 붙는다.
- 클라이언트 화면에 보이는 것만 쓴다. 상대 팀의 포지션은 세션에 없으므로 챔피언별 포지션 통계로 **추정**하고, 화면에 추정이라고 표시한다.

## Riot API — 사용자가 넣은 자기 키로

키가 없으면 아래는 하나도 부르지 않는다. 추천 화면은 키 없이 돈다.

서버는 `0_Setup.cmd` · `Settings.cmd`에서 고른다 (기본 한국). 아래 `kr` · `asia` 자리는 고른 서버에 따라 바뀐다 —
예: 유럽 서부면 `euw1` · `europe`, 북미면 `na1` · `americas`, 동남아면 `sg2` · `sea` (계정 조회만 `asia`). `src/region.py`.

| 엔드포인트 | 기능 |
|---|---|
| `kr` `/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}` | 수집 — 티어를 아는 플레이어 목록 (아이언~플래티넘) |
| `kr` `/lol/league/v4/{master,grandmaster,challenger}leagues/by-queue/RANKED_SOLO_5x5` | 고티어 수집 (`advanced\HighTierCollect.cmd`, 실험용) |
| `asia` `/lol/match/v5/matches/by-puuid/{puuid}/ids` | 수집 · 내 기록 — 최근 랭크 매치 ID |
| `asia` `/lol/match/v5/matches/{matchId}` | 수집 · 내 기록 — 경기 결과 |
| `asia` `/lol/match/v5/matches/{matchId}/timeline` | 내 기록 — 승패 육각형의 라인전 축 · 기준 데이터 (`advanced\Benchmark.cmd`) |
| `asia` `/riot/account/v1/accounts/by-riot-id/{gameName}/{tagLine}` | 내 기록 — 사용자가 `my_accounts.txt`에 적은 자기 Riot ID를 찾는다 |

- 개발용 키 한도(초당 20회 · 2분 100회)보다 낮게 스스로 속도를 맞춘다.
- 수집한 경기는 사용자 PC의 `data/matches.db`에만 저장되고 어디에도 올리지 않는다. 플레이어 puuid는 다음 매치 ID를 찾는 데만 쓴다.
- 내 기록에서 다른 플레이어의 이름 · 태그 · puuid는 저장하기 전에 지운다.

## 그 밖

| 주소 | 무엇 |
|---|---|
| `ddragon.leagueoflegends.com` | 챔피언 목록 · 이미지 (키 없음) |
| `api.github.com/repos/taejogmin/PickNote/releases` | `Update.cmd` — 새 버전이 있는지 |
| `astral.sh` | `0_Setup.cmd` — 파이썬 설치 도구 uv (처음 한 번) |

---

## Riot 개발자 포털 등록용 설명 (영어)

> **PickNote** is a free, open-source desktop tool (MIT) for League of Legends Solo/Duo players (Iron–Platinum; Korea by default, other servers selectable). During champion select it shows which champions raise the team's win rate in the player's role, with the reasons (champion strength, ally synergy, enemy counters), from a model trained on aggregated ranked match results. It never picks, bans or changes anything for the player and offers several options rather than a single answer.
>
> **League Client API (read-only, localhost only):** `GET /lol-champ-select/v1/session`, polled once per second while the app is open, to read the picks, hovers, bans and the player's assigned position that are already visible in the client. No write endpoints are used. Enemy positions are inferred from public per-champion role statistics and clearly marked as guesses.
>
> **Riot API (only with the user's own key, for the user's own use):** league-v4 entries (seed players by tier), match-v5 matchlists by PUUID, match-v5 matches and timelines, and account-v1 by Riot ID (to fetch the user's own match history). Data is stored only on the user's PC and is never redistributed; other players' names and PUUIDs are removed from the user's match history before storage. The distributed package contains only a trained model (aggregate statistics), not match data.

# PickNote

**[한국어](README.ko.md)** · English

A ban/pick helper for League of Legends **Solo/Duo, Iron–Platinum**, and an assistant that gives feedback from your own match stats.
It's built to run on your own PC — don't run it on shared or public computers, where your API key and accounts could leak.

While you're in champion select, it reads the draft from your client and shows which champions
give your team the best win rate in your role — and *why* (champion strength, synergy with allies, counters against enemies).

- Learned from about **1.7 million real KR ranked games** (Iron–Platinum, patches 16.15–16.21), not a tier list
- Works with a **partial draft** — first pick, one enemy revealed, anything in between
- Reads the client **read-only**. It never picks or bans for you, and doesn't touch spells or runes
- Korean / English UI
- Korea by default. If you play on another server, pick it during setup

## Quick start (Windows)

1. Download the latest **Source code (zip)** from [Releases](https://github.com/taejogmin/PickNote/releases) and **extract it** (right-click → Extract All).
2. Double-click **`0_Setup.cmd`**. It installs Python and everything else inside the folder — no separate Python install needed. Pick your language and server (Enter = Korea).
3. Double-click **`3_Open.cmd`**. The app opens in your browser. Start a game and enter champion select — the draft fills in on its own.

That's all you need for recommendations. The steps below are optional.

| File | What it does |
|---|---|
| `0_Setup.cmd` | First time only. Installs Python + packages, language, server |
| `1_ApiKey.cmd` | Opens the Riot developer site and Notepad so you can paste your API key |
| `2_Collect.cmd` | Collects new games from your server (needs an API key). Leave it running |
| `3_Open.cmd` | Opens the app |
| `4_MyRecord.cmd` | Opens Notepad for your Riot ID(s), then downloads your games for the **My record** page |
| `Update.cmd` | New app version (asks first) · retrains on new games · refreshes your record |
| `Settings.cmd` | Change language or server |
| `Status.cmd` | How much has been collected, whether your key still works |
| `advanced\` | High-tier collection, autostart at sign-in, forced retrain, hexagon baselines |

### API key (for collecting and *My record*)

Collecting games and downloading your own record need a free **Riot development API key**.
Run `1_ApiKey.cmd`: sign in at developer.riotgames.com, press *Regenerate API Key*, and paste the `RGAPI-…` key into the Notepad window that opens.
Development keys expire every 24 hours. The collector doesn't stop when the key expires — paste a new one and it carries on.

### My record and servers

`4_MyRecord.cmd` opens Notepad. Write your Riot ID (`Name#Tag`), one per line, and save.
For an account on a different server than your setting, add `@server` after it, e.g. `Name#EUW @euw1`.

Server codes: `kr` `jp1` `na1` `br1` `la1` `la2` `euw1` `eun1` `tr1` `ru` `me1` `oc1` `sg2` `tw2` `vn2`

### Keeping it fresh

Run `Update.cmd` now and then. It:

1. checks for a new version and **asks before installing** (your key, accounts, data and models are kept),
2. **retrains** once you've collected enough games — if the new model scores worse than the current one, the current one is kept,
3. downloads new games for the accounts in `my_accounts.txt`.

**When does it retrain?** Collected games aren't shared (see *Data and privacy*), so you start with the bundled model.
Once you've collected **300,000 games** it trains on them for the first time, then again every 30,000 new games (10,000 on a new patch).
A development key collects about 50,000 games a day.

**Requirements:** retraining needs **8 GB of RAM or more** (16 GB is comfortable). On ~2 million games it took 16–17 minutes and peaked at about 5.2 GB. Recommendations alone need very little.

**Other servers:** the bundled model was trained on Korean games. On another server, the app uses it until you've collected 300,000 games there, and says so at the top of the screen.

## How it works

[docs/how-it-works.md](docs/how-it-works.md) (Korean) · the full list of endpoints it calls: [docs/api-usage.md](docs/api-usage.md)

In short: a factorization-machine model over (champion, position) slots, with separate vectors for ally synergy and enemy counters,
validated on the most recent 8 days of games it never trained on.

## Data and privacy

- **Collected match data is not published.** Riot's API terms don't allow passing game data on to others. Only the model trained on those games (aggregate values) is shared; everyone collects games with their own key.
- Collected games stay in `data/` on your PC.
- *My record* stays in `data/me.db` on your PC. Other players' names and IDs are removed before saving; teammates are kept only as a salted hash so repeat duos can be recognized.
- Your API key stays in `.env` on your PC and is only sent to Riot.

## License

[MIT](LICENSE)

PickNote isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.

# Supply Scenario App

A Streamlit application for comparing supply scenarios.

## If reboot still shows the old app (most common fix)

A **reboot does not pull new code**. It only restarts the current deployed commit.

Use this exact sequence:

1. Open GitHub repo: `https://github.com/mbf2188-rgb/supply-scenario-app`
2. Click **Pull requests**.
3. If you see a PR for branch `codex-update` (or `work`), open it.
4. Click **Merge pull request** → **Confirm merge**.
5. Go back to Streamlit Community Cloud and click **Reboot app**.

If there is no PR, follow "Create PR in 3 commands" below.

## Why a `work` branch existed

A `work` branch is a temporary **feature branch**. It keeps in-progress changes separate from `main`.

Your Streamlit app was deployed from `main`, so changes on `work` were invisible until merged.

## Create PR in 3 commands (copy/paste)

Run in terminal at repo root:

```bash
git checkout codex-update
git push -u origin codex-update
```

Then open this link and click **Create pull request**:

`https://github.com/mbf2188-rgb/supply-scenario-app/compare/main...codex-update`

On the PR page:
1. Click **Create pull request**.
2. Click **Merge pull request**.
3. Click **Confirm merge**.


## If GitHub says "nothing to compare"

That message means the two branches are already identical.

For your screenshot (`base: main`, `compare: codex-update`), this means **there is no pending PR to merge**.

Next step is deployment targeting, not Git merging:
1. In Streamlit, verify app points to `mbf2188-rgb/supply-scenario-app`, branch `main`, file `app.py`.
2. Confirm the app URL is the app tied to that repo (not an older duplicate app).
3. Reboot app.

## Run locally (simple workflow)

Use only `main` locally:

```bash
git checkout main
git pull origin main
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud settings to verify

Open the app card and confirm it shows:
- Repository: `mbf2188-rgb/supply-scenario-app`
- Branch: `main`
- Main file: `app.py`

If Branch is not `main`, create a new app pointing to `main` and `app.py`.

## Quick checks you can run in terminal

```bash
git branch --show-current
git log --oneline -n 3
```

Expected after merge:
- current branch = `main`
- top commit includes the latest app changes


## If `lib/maps.py` is missing in your terminal

If you run a command and get `FileNotFoundError: lib/maps.py`, your checked-out code is an older single-file layout.

Run these exact commands first:

```bash
pwd
ls -la
find . -maxdepth 2 -type f | sort
```

Then run this to show the app version you have:

```bash
python - <<'PY2'
from pathlib import Path
p = Path('app.py')
print('app.py exists:', p.exists())
if p.exists():
    text = p.read_text()
    print('has Build caption:', 'Build:' in text)
    for key in ['Assigned Terminal', 'Global Controls', 'Scenario to display', 'Baseline scenario']:
        print(f"contains {key!r}:", key in text)
PY2
```

If `contains 'Assigned Terminal': True` and `contains 'Scenario to display': False`, you are running the old app code.
At that point, do **not** run the `lib/maps.py` patch commands; they are for the modular layout only.

## Quick troubleshooting

- App looks old after merge: hard refresh browser (`Ctrl+Shift+R` / `Cmd+Shift+R`) then reboot app.
- `localhost` does not open: you are in a remote environment; use that environment's forwarded URL.


## One-shot fix for legacy single-file `app.py`

If your repo has only `app.py` + `requirements.txt` (no `lib/` folder), run:

```bash
python scripts/legacy_hotfix_single_file.py
git add app.py scripts/legacy_hotfix_single_file.py
git commit -m "Apply legacy app map + build hash hotfix"
git push origin main
```

Then reboot Streamlit app.

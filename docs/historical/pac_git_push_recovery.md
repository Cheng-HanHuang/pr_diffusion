# PAC Git push recovery (diverged `main`)

If you see:

- `git push` rejected with `fetch first`, and
- `git pull` says `Need to specify how to reconcile divergent branches`,

you have local commits on `main` and remote `origin/main` also advanced.

This guide gives a safe recovery path.

---

## 0) Quick safety backup (always do first)

```bash
git branch backup/pac-before-recovery
```

---

## 1) Set pull behavior once

Recommended (linear history):

```bash
git config pull.rebase true
```

(Use `--global` if you want this for all repos.)

---

## 2) Keep your local commit but move it off `main`

Assume your local commit hash is `b483dd8`.

```bash
# from repo root
git switch -c pac/local-upload-check

# update from remote
git fetch origin
git switch main
git reset --hard origin/main

# return to your working branch and replay the commit on top of latest main
git switch pac/local-upload-check
git rebase main
```

Now your `main` is clean and your PAC commit is on a dedicated branch.

---

## 3) Decide what to keep from that commit

For this project, prefer **not** committing bulky local env snapshots and temporary script copies.

Usually keep:

- reproducibility docs,
- small manifest files,
- lightweight summaries.

Usually drop from Git history:

- `conda_list.txt`, `pip_freeze.txt`, `env_full.yml`,
- temporary wrappers like `scripts/pr_*` (if they are local-only copies).

If needed, split into two commits:

```bash
# unstage everything first
git reset

# stage only files you really want to keep
git add <wanted_files>
git commit -m "Keep reproducibility metadata only"
```

---

## 4) Push safely

```bash
git push -u origin pac/local-upload-check
```

Open a PR from `pac/local-upload-check` into `main`.

---

## 5) If you want to discard the local upload commit entirely

```bash
git switch main
git fetch origin
git reset --hard origin/main
```

This removes local unpushed commits from `main` (but your backup branch from step 0 still preserves them).

---

## 6) Optional identity fix

Set explicit identity so future commits do not show the warning:

```bash
git config --global user.name "Cheng-Han Huang"
git config --global user.email "<your-email>@msu.edu"
```

To fix author on the latest commit:

```bash
git commit --amend --reset-author
```

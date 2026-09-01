# Contributing to Tiny Hybrid RAG

This project uses a lightweight, short-lived branch workflow.

## Branch roles

- `main` is the stable integration branch.
- New work starts from an up-to-date `main` on one focused branch.
- Short-lived branches are reviewed and validated through a Pull Request before they enter `main`.
- Finished branches are deleted after merging; we do not maintain long-lived `develop`, `release`, or `staging` branches.

Use these prefixes:

- `feat/` — application features
- `infra/` — Docker, Compose, CI, deployment, and infrastructure
- `fix/` — bug fixes
- `chore/` — tooling, configuration, and maintenance
- `docs/` — documentation-only changes

Examples of future branch names include:

```text
feat/hybrid-retrieval
infra/docker-compose
chore/git-precheck
fix/chroma-persistence
```

Only create a branch when there is real work for it. Keep each branch focused on one coherent change.

## Workflow

```text
stable main
    ↓ create <type>/<topic>
develop and test locally
    ↓ stage + git precheck + commit focused changes
push the feature branch
    ↓ open Pull Request to main
human review + CI evidence
    ↓ merge only after validation
delete the merged feature branch
```

A typical local start is:

```bash
git switch main
git pull --ff-only
git switch -c <type>/<topic>
```

Install the repository-local convenience alias once per clone:

```bash
scripts/setup-git-alias.sh
```

Git cannot distribute `.git/config` aliases through committed files, so the setup script makes the local change explicitly and never touches global Git configuration.

Before committing, stage the intended change and inspect it:

```bash
git add .
git precheck
git commit -m "chore: describe the focused change"
git push -u origin HEAD
```

`git precheck` is a fast filename-based warning system. It does not run tests or replace GitHub Actions CI.

## Commit and Pull Request scope

Use concise commit messages that describe the milestone or behavior being changed. A Pull Request should explain its purpose, the checks performed, and any surprising results. Generated Chroma data, environments, caches, credentials, and other ignored local state must not be committed.

After CI succeeds, prefer merging through the GitHub Pull Request. If a local merge is appropriate, the full manual lifecycle is:

```bash
git switch main
git pull --ff-only
git merge <type>/<topic>
git push
git branch -d <type>/<topic>
git push origin --delete <type>/<topic>
```

Local and remote branch deletion are separate. Deleting a merged branch pointer does not delete commits already reachable from `main`.

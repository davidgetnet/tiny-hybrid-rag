# Contributing to Tiny Hybrid RAG

This project uses a lightweight, short-lived feature-branch workflow.

## Branch roles

- `main` is the stable integration branch.
- New work starts from an up-to-date `main` on a focused branch named `feature/<topic>`.
- Feature branches are reviewed and validated through a Pull Request before they enter `main`.
- Feature branches are deleted after merging; we do not maintain long-lived `develop`, `release`, or `staging` branches.

Examples of future branch names include:

```text
feature/graph-retrieval
feature/hybrid-search
feature/langgraph
```

Only create a branch when there is real work for it. Keep each branch focused on one coherent change.

## Workflow

```text
stable main
    ↓ create feature/<topic>
develop and test locally
    ↓ commit focused changes
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
git switch -c feature/<topic>
```

Before proposing a change, run the relevant tests and inspect the diff. Future GitHub Actions will repeat tests and Docker validation remotely, but local responsibility does not disappear when automation is added.

## Commit and Pull Request scope

Use concise commit messages that describe the milestone or behavior being changed. A Pull Request should explain its purpose, the checks performed, and any surprising results. Generated Chroma data, environments, caches, credentials, and other ignored local state must not be committed.

There is currently no automated CI or deployment configuration. Those layers will be added in later learning increments.

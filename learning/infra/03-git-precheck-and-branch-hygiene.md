# Infrastructure Increment 03: Git Precheck and Branch Hygiene

This increment adds a fast local warning system around the Git workflow. It does not change Tiny RAG behavior, run the full test suite, or add another RAG phase.

The desired everyday loop is:

```text
make change
    ↓
git add .
    ↓
git precheck
    ↓
understand branch + CI risk
    ↓
commit → push → GitHub Actions → merge → delete finished branch
```

## Two related responsibilities

Git records and organizes repository history:

```text
main
 ↓ create short-lived branch
branch commits
 ↓ review and merge
main contains accepted commits
 ↓
delete finished branch pointer
```

Automated verification gathers evidence about a proposed change:

```text
local precheck
 ↓ quick warnings
push
 ↓
GitHub Actions
 ↓ unit tests + integration test + Docker build
CI result
```

They are related because a branch carries a change into CI, but they are not the same. Git can preserve broken code perfectly. CI can test a commit without deciding how branches should be organized.

## The lightweight branch convention

`main` is the accepted and stable integration branch. New work belongs on one short-lived branch with a name that communicates its purpose:

```text
feat/    application feature
infra/   Docker, Compose, CI, deployment, infrastructure
fix/     bug fix
chore/   tooling, configuration, maintenance
docs/    documentation only
```

Examples for this repository are `feat/hybrid-retrieval`, `infra/docker-compose`, `fix/chroma-persistence`, and `chore/git-precheck`.

There is no permanent `develop`, `release`, or staging branch. A small repository does not benefit from the coordination cost of GitFlow.

## Why short-lived branches?

A branch represents one isolated unit of unfinished work. It lets the developer compare that work with `main`, push it without declaring it accepted, run CI, and discuss it in a Pull Request.

The branch has completed its purpose once its commits are accepted into `main`. Keeping many finished branch names does not preserve extra code; it preserves obsolete labels.

## A branch is a pointer

Commits form history. A branch name is primarily a movable pointer to one commit:

```text
main ───────────────► accepted commit
                       ▲
feat/something ────────┘
```

After the feature is merged, both names may point into history already reachable from `main`. Deleting `feat/something` deletes that pointer, not the merged commits.

This is why branch cleanup is normally safe only after checking that the branch is merged. The precheck reports candidates but never deletes them.

## What `git precheck` inspects

The script reports:

- the current and base branches;
- ahead/behind counts when `main` or `origin/main` is available;
- working-tree state;
- staged filenames;
- unstaged and untracked filenames;
- staged change categories;
- simple CI-alignment warnings;
- local branches already merged into `main`.

Its categories are filename heuristics:

```text
requirements*.txt / pyproject.toml  → dependency/runtime
Dockerfile                          → Docker
compose.yml / docker-compose.yml    → Compose
tests/                              → tests
.github/workflows/                  → CI workflow
src/ / data/ / infra/ / .env.example → application/runtime configuration
```

Filename inspection can ask a useful question, but it cannot answer whether code is correct. A dependency change without a CI YAML change might be perfectly aligned when CI already installs `requirements-dev.txt`. The precheck therefore prints a warning to verify the relationship rather than declaring an error.

## INFO, WARNING, and ERROR

`INFO` explains state or suggests a normal next step. For example, no staged files means there is no intended commit to classify.

`WARNING` identifies something worth reviewing: dependencies changed, container configuration changed, or application code changed without staged tests. Warnings do not block a commit.

`ERROR` is reserved for a condition that prevents the script from doing its basic job, such as running outside a Git repository.

The script exits successfully when it finds warnings. It is a thinking aid, not a policy enforcement hook.

## Main protection without automation

When meaningful application or infrastructure files are staged on `main`, the script prints:

```text
WARNING: You are on main with staged implementation or infrastructure changes.
Consider creating a short-lived branch before committing:
    git switch -c <type>/<name>
```

It does not create the branch or block the commit. Documentation-only edits do not trigger the implementation warning.

## Installing the local alias

Git aliases live in Git configuration, not ordinary tracked repository files. A committed alias cannot automatically appear in another clone's `.git/config`.

Run this once per clone:

```bash
scripts/setup-git-alias.sh
```

The setup script executes the equivalent of:

```bash
git config --local alias.precheck '!scripts/precheck.sh'
```

`--local` writes only this repository clone's `.git/config`. It does not silently modify global user configuration. The everyday command then becomes:

```bash
git add .
git precheck
```

The script remains directly runnable as `scripts/precheck.sh`, including before the alias is installed.

## Expected developer workflow

Start one focused increment:

```bash
git switch main
git pull --ff-only
git switch -c infra/docker-compose

# make changes
git add .
git precheck
git commit -m "infra: add Docker Compose setup"
git push -u origin infra/docker-compose
```

Prefer review and merging through a GitHub Pull Request. If a local merge is intentionally used after CI succeeds:

```bash
git switch main
git pull --ff-only
git merge infra/docker-compose
git push
git branch -d infra/docker-compose
git push origin --delete infra/docker-compose
```

`git branch -d` removes the local pointer. `git push origin --delete` removes the remote pointer. They are separate operations, and neither is performed by the precheck.

## Precheck versus CI

```text
git precheck
    ↓
fast local filename and branch warning system

GitHub Actions CI
    ↓
actual automated unit, integration, and image-build verification
```

`git precheck` is not CI, a replacement for tests, proof that Tiny RAG works, or proof that `.github/workflows/ci.yml` is correct. It is intentionally fast enough to run before most commits and catches obvious omissions before spending a remote CI run.

The current CI remains the authority for executable checks. It installs declared dependencies, runs existing Python checks, exercises Chroma over HTTP, and builds the Docker image on a clean Ubuntu runner.

## Repository situation observed in this increment

The work began with a clean tree on `feature/manual-graph-retrieval`, which was not yet merged into `main`. The tooling work was therefore isolated on `chore/git-precheck` based on `main` instead of mixing two increments.

At inspection time, these local branches appeared merged into `main`:

```text
ci/github-actions-foundation
feature/knowledge-graph
infra/docker-compose-chroma-ui
```

They are candidates for human-reviewed cleanup, not automatic deletion targets. `feature/hybrid-retrieval` and `feature/manual-graph-retrieval` were not merged into `main` and must not be presented as cleanup candidates.

The remote `feature/knowledge-graph` pointer was already gone while the local pointer remained, illustrating that local and remote branch state are separate.

## What to do when the warning is correct

If application code changes without tests, first ask whether an existing test already covers the behavior. If it does, no artificial test edit is needed. If the behavior is new and important, add a meaningful test.

If requirements change, check how `requirements-dev.txt`, Docker, and CI install dependencies. A workflow edit is necessary only when the installation contract changes.

If Docker or Compose changes, check whether the existing `docker-build` and integration jobs still exercise the affected path. The goal is alignment, not changing CI YAML every time an infrastructure filename changes.

The broader lesson is that repository tooling should make good judgment easier without pretending to replace it.

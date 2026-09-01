#!/bin/sh

# Fast, filename-based Git and CI-alignment hints. This does not run CI.
set -u

info() {
    printf 'INFO: %s\n' "$1"
}

warning() {
    printf 'WARNING: %s\n' "$1"
}

error() {
    printf 'ERROR: %s\n' "$1" >&2
}

yes_no() {
    if [ "$1" -eq 1 ]; then
        printf 'YES'
    else
        printf 'NO'
    fi
}

matches_any() {
    patterns=$1
    printf '%s\n' "$staged_files" | grep -Eq "$patterns"
}

if ! repo_root=$(git rev-parse --show-toplevel 2>/dev/null); then
    error "Not inside a Git repository."
    exit 2
fi

cd "$repo_root" || exit 2

current_branch=$(git branch --show-current)
if [ -z "$current_branch" ]; then
    current_branch="(detached HEAD)"
fi

if git show-ref --verify --quiet refs/heads/main; then
    base_ref=main
elif git show-ref --verify --quiet refs/remotes/origin/main; then
    base_ref=origin/main
else
    base_ref=""
fi

status_porcelain=$(git status --short)
staged_files=$(git diff --cached --name-only --diff-filter=ACMRD)
unstaged_files=$(git diff --name-only --diff-filter=ACMRD)
untracked_files=$(git ls-files --others --exclude-standard)

if [ -z "$status_porcelain" ]; then
    working_tree="clean"
elif [ -n "$staged_files" ]; then
    working_tree="modified (staged changes present)"
else
    working_tree="modified (nothing staged)"
fi

printf '\nGIT CONTEXT\n'
printf '%s\n' '-----------'
printf 'Current branch: %s\n' "$current_branch"
printf 'Base branch: main\n'

if [ -n "$base_ref" ]; then
    counts=$(git rev-list --left-right --count "$base_ref"...HEAD 2>/dev/null || printf '')
    if [ -n "$counts" ]; then
        behind=$(printf '%s' "$counts" | awk '{print $1}')
        ahead=$(printf '%s' "$counts" | awk '{print $2}')
        printf 'Ahead of %s: %s commit(s)\n' "$base_ref" "$ahead"
        printf 'Behind %s: %s commit(s)\n' "$base_ref" "$behind"
    else
        warning "Could not determine ahead/behind counts."
    fi
else
    warning "Neither local main nor origin/main exists; ahead/behind is unavailable."
fi

printf 'Working tree: %s\n' "$working_tree"

printf '\nSTAGED FILES\n'
printf '%s\n' '------------'
if [ -n "$staged_files" ]; then
    printf '%s\n' "$staged_files"
else
    printf '(none)\n'
fi

printf '\nUNSTAGED / UNTRACKED FILES\n'
printf '%s\n' '--------------------------'
if [ -n "$unstaged_files" ]; then
    printf '%s\n' "$unstaged_files"
fi
if [ -n "$untracked_files" ]; then
    printf '%s\n' "$untracked_files" | sed 's/^/? /'
fi
if [ -z "$unstaged_files" ] && [ -z "$untracked_files" ]; then
    printf '(none)\n'
fi

dependency_change=0
docker_change=0
compose_change=0
tests_change=0
ci_change=0
application_change=0

if [ -n "$staged_files" ]; then
    matches_any '(^|/)(requirements[^/]*\.txt|pyproject\.toml|poetry\.lock|Pipfile(\.lock)?)$' && dependency_change=1
    matches_any '(^|/)(Dockerfile|Containerfile)(\..*)?$|(^|/)(dockerfile|containerfile)$' && docker_change=1
    matches_any '(^|/)(compose|docker-compose)\.ya?ml$' && compose_change=1
    matches_any '^tests/' && tests_change=1
    matches_any '^\.github/workflows/' && ci_change=1
    matches_any '^src/|^infra/|^data/|(^|/)\.env\.example$' && application_change=1
fi

printf '\nSTAGED CHANGE CATEGORIES\n'
printf '%s\n' '------------------------'
printf 'Dependency/runtime change: %s\n' "$(yes_no "$dependency_change")"
printf 'Docker change: %s\n' "$(yes_no "$docker_change")"
printf 'Compose change: %s\n' "$(yes_no "$compose_change")"
printf 'Tests changed: %s\n' "$(yes_no "$tests_change")"
printf 'CI workflow changed: %s\n' "$(yes_no "$ci_change")"
printf 'Application/runtime configuration changed: %s\n' "$(yes_no "$application_change")"

printf '\nALIGNMENT HINTS\n'
printf '%s\n' '---------------'
if [ -z "$staged_files" ]; then
    info "No staged files to classify. Stage the intended commit, then rerun git precheck."
else
    [ "$dependency_change" -eq 1 ] && warning "Dependencies changed. Verify that CI installs dependencies in a way that picks up this change."
    if [ "$docker_change" -eq 1 ] || [ "$compose_change" -eq 1 ]; then
        warning "Container/infrastructure configuration changed. Check whether CI still tests the real build/runtime path."
    fi
    if [ "$application_change" -eq 1 ] && [ "$tests_change" -eq 0 ]; then
        warning "Application or runtime code changed without staged test changes. Confirm existing tests already cover this behavior."
    fi
    [ "$ci_change" -eq 1 ] && info "CI workflow changed. Review the YAML and confirm its commands remain locally reproducible."
    if [ "$dependency_change" -eq 0 ] && [ "$docker_change" -eq 0 ] && \
       [ "$compose_change" -eq 0 ] && [ "$ci_change" -eq 0 ] && \
       [ "$application_change" -eq 0 ]; then
        info "No obvious CI-alignment warning found from the staged filenames."
    fi
fi

meaningful_change=0
if [ "$dependency_change" -eq 1 ] || [ "$docker_change" -eq 1 ] || \
   [ "$compose_change" -eq 1 ] || [ "$ci_change" -eq 1 ] || \
   [ "$application_change" -eq 1 ]; then
    meaningful_change=1
fi

if [ "$current_branch" = "main" ] && [ "$meaningful_change" -eq 1 ]; then
    printf '\n'
    warning "You are on main with staged implementation or infrastructure changes."
    printf 'Consider creating a short-lived branch before committing:\n'
    printf '    git switch -c <type>/<name>\n'
fi

printf '\nMERGED LOCAL BRANCHES\n'
printf '%s\n' '---------------------'
if [ -n "$base_ref" ]; then
    merged_branches=$(git for-each-ref --merged="$base_ref" --format='%(refname:short)' refs/heads/ 2>/dev/null | \
        awk -v current="$current_branch" '$0 != "main" && $0 != current')
    if [ -n "$merged_branches" ]; then
        printf '%s\n' "$merged_branches"
        info "These appear merged into $base_ref and may be deletion candidates after review."
    else
        printf '(none)\n'
    fi
else
    printf '(unavailable: main was not found)\n'
fi

printf '\nMANUAL BRANCH CLEANUP\n'
printf '%s\n' '---------------------'
printf 'Inspect merged local branches: git branch --merged main\n'
printf 'Delete one merged local branch: git branch -d <branch>\n'
printf 'Delete its remote pointer separately: git push origin --delete <branch>\n'
printf 'No branch was changed or deleted by this script.\n'

printf '\nPRECHECK LIMIT\n'
printf '%s\n' '--------------'
info "Filename heuristics are warnings, not proof. GitHub Actions CI remains the automated verification authority."

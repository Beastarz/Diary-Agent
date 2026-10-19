# commit-and-push

Stage everything in the working tree, group changes into logical commits, and push the branch to origin.

## Usage

```
/commit-and-push
```

No arguments needed. The skill reads the working tree and figures out the rest.

---

## Steps

### 1. Read config

Read `.claude/work/config.json` to get `repo` and `gh_host`.

### 2. Inspect the working tree

```bash
git status --short
git diff --stat HEAD
```

Also read untracked files (`??` in `git status`) to understand their content — they are invisible to `git diff HEAD`.

Collect the full list of changed/added/deleted/untracked files.

### 3. Determine commit grouping

**Default: one commit.**

Split into multiple commits only when the changes are genuinely independent and a single message would have to list unrelated concerns. The bar is high — two or three commits maximum per push.

Rules:

- Changes to the same feature area → one commit
- If a single subject line would need to say "X, Y, and Z" where X, Y, Z are **unrelated** subsystems → split
- Test files that accompany their source file → same commit as the source
- Config / tooling changes that only exist to support the feature → same commit as the feature
- Never split a file from the tests that cover it

### 4. Draft commit message(s)

Format (from `work-artifacts` contract):

```
{type}({scope}): {description} (#{issue_number})
```

- **type**: `feat` · `fix` · `chore` · `refactor` · `perf` · `docs` · `test` · `ci`
- **scope**: optional; primary area changed (`chat`, `api`, `auth`, `ui`, `skills`, etc.)
- **description**: imperative, lowercase, no trailing period, ≤72 chars total
- **issue number**: extract from the current branch name (`{type}/{issue_number}-{slug}`)

To extract the issue number from the branch:

```bash
git rev-parse --abbrev-ref HEAD
# e.g. chore/1042-add-start-task-skill → issue #1042
```

If the branch name contains no issue number (hotfix, detached HEAD, etc.), omit the `(#N)` suffix.

### 5. Stage and commit

**Before staging anything**, scan the file list collected in step 2 for sensitive filenames (`.env`, `*.key`, `credentials*`, `default-env.json`). If any are found, stop and ask the user before running any `git add`. Only proceed once confirmed safe.

For **one commit** — stage everything and commit:

```bash
git add -A
git commit -m "{message}"
```

For **multiple commits** — stage file by file per group, then commit each:

```bash
git add <file1> <file2> ...
git commit -m "{message for group 1}"

git add <file3> <file4> ...
git commit -m "{message for group 2}"
```

### 6. Push

```bash
git push origin HEAD
```

If the push is rejected because the remote branch doesn't exist yet, use:

```bash
git push --set-upstream origin HEAD
```

If the push is rejected due to non-fast-forward (remote has commits the local branch doesn't), **stop and tell the user** — do not force-push.

### 7. Report

Tell the user:

- Each commit hash + message (from `git log --oneline -N`)
- The branch that was pushed
- A link to the branch on GitHub: https://{gh_host}/{repo}/tree/{branch}

# Git directory skeleton plan

Date: 2026-09-21

Purpose: retain the existing formal project skeleton without tracking raw data,
scientific results, or large runtime artifacts. This is an engineering
maintenance record; no research task, task move, or Git operation is performed
by this plan.

## Create .gitkeep

| Directory | Exists now | Trackable files before this change | Decision | Reason |
|---|---:|---:|---|---|
| data/raw/scenario_1/ | yes | no | create | canonical raw scenario root |
| data/raw/scenario_2/ | yes | no | create | canonical raw scenario root; experiment data remains ignored |
| data/processed/scenario_1/ | yes | no | create | canonical processed-data scenario root |
| data/processed/scenario_2/ | yes | no | create | canonical processed-data scenario root |
| results/<each of the 10 fixed modules>/ | yes | no | create | fixed module roots; actual results remain ignored |
| results/behavior_modeling/scenario_2/ | yes | no | create | existing formal scenario root |
| results/behavior_fingerprint_retrieval/scenario_2/ | yes | no | create | existing formal scenario root |
| results/pa_performance_evaluation/cross_scenario/ | yes | no | create | existing formal cross-scenario root |
| results/retrieval_oriented_model_selection/self_hit_oriented/scenario_2/ | yes | no | create | existing route/scenario root |
| results/retrieval_oriented_model_selection/dpd_shareability_oriented/scenario_2/ | yes | no | create | existing route/scenario root |
| work_logs/retrieval_oriented_model_selection/dpd_shareability_oriented/scenario_2/ | yes | no | create | existing route/scenario root with no tracked file; detailed search state remains local |

The exact files created by this plan are the empty .gitkeep files at the
directories above. No task-level .gitkeep is planned.

## Skip .gitkeep

- scripts/ module and task directories: existing __init__.py or source files already preserve them.
- work_logs/ module and existing task/scope directories that already contain tracked text logs.
- results/ task directories: they are scientific output directories, not permanent Git skeleton nodes.
- data/raw/scenario_2/experiment_2026_0816/ and its subdirectories: actual experiment data remains ignored.
- cross_scenario/ under data/raw/ or data/processed/: not part of the canonical data layout.
- Any future, empty, or invented task/experiment directory.

## Invariants

- data/raw and data/processed real data remain ignored.
- results real scientific outputs remain ignored.
- work_logs is not globally ignored; text logs remain reviewable.
- Candidate caches, checkpoints, screening and runtime directories are ignored narrowly.
- environment.yml, pyproject.toml, .gitattributes, scientific source, results, checkpoint/cache contents and raw data are not modified by this maintenance task.
- No git add, git commit, git push, git reset, git restore, git clean, git checkout, git switch, git stash, git rm, or git rm --cached is performed.

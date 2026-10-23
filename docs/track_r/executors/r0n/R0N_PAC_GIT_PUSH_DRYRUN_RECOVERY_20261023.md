# R0-N PAC dry-run push blocker — safe recovery (2026-10-23)

## Evidence (operator log)

The first comprehensive launcher, pinned to `07d2529ae09a8cff2f68cf354e402fb75932a21f`, stopped **before** its combined controller started:

```
Warning: you are leaving 1 commit behind:
 d4c80d4 track-r R0N N0: archive SHA-verified R0.8 aggregate tables
HEAD is now at 07d2529 ...
STOP_R0N_GIT_PUSH_PERMISSION_OR_REMOTE_DIVERGENCE
R0N_LAUNCH_RC=6
```

The original N0 commit was a local detached commit from an earlier
attempt; checking out the implementation left it unattached. The
remote branch was subsequently checked through GitHub and was
**still exactly `07d2529` at the time**: no remote
divergence existed then. The first launcher redirected all push
stderr, so the *precise* reason for the dry-run failure
(authentication, permission, transport, or policy) is unknown.
Do not claim credentials were definitively the cause.

## Remediation implemented

- Git dry-run now reports the last six diagnostic lines instead of
  hiding stderr or blocking otherwise-valid science.
- Original read-only R0.8 ZIP SHA and per-member SHA remain mandatory.
- The combined controller first creates **local** Git commits and
  tries a normal, non-force push for up to 30 seconds. A push
  failure is recorded in `push_errors`; N1/N2/N3 proceed on
  the local commit, and the planner ZIP is produced.
- Final state appends `_LOCAL_PUSH_PENDING` when GitHub has not
  received the commits. Do not claim remote publication on
  that state. The local HEAD and ZIP remain recoverable.
- `GIT_TERMINAL_PROMPT=0` prevents auth prompts from hanging nohup.
- The user's N0 commit `d4c80d4` should first be anchored with
  a local backup branch, preserving all its original archive files.
  The comprehensive job can then import the same SHA-pinned aggregate
  ZIP afresh from a newer, clean implementation HEAD without
  cherry-picking across implementation changes or force-pushing.
- Never run `git reset --hard`, `git push --force`, `git clean -fd`,
  or Git garbage collection to recover this scenario.

## Remaining work

R0-N stages N0, N1, optional N2 and N3 have **not** been shown
complete in the operator logs. The next PAC attempt must pass
full CPU suite, exact ZIP SHA, K3 replay, 3,240 score receipt
validation, the 60-config grid and N2 CLIP runtime ceiling.
The science and frozen planner boundaries are unchanged:
no R1, no new reconstruction or measurement, no protected
B24 confirmation305 access.

Only `R0N_ALL_STAGES_COMPLETE_PLANNER_READY` or
`R0N_PARTIAL_OPTIONALS_PLANNER_READY` denotes a local scientific
return with a successful remote push; corresponding
`_LOCAL_PUSH_PENDING` statuses denote local return only.

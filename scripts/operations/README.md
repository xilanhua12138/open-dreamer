# PAI-DSW operations

`poll_pai_dsw_dynamics.py` is the capacity and launch owner for the existing
CoinRun A10. It is deliberately a one-shot program: macOS `launchd` invokes it
every five minutes, and an advisory lock prevents overlapping invocations.

Safety invariants:

- It only addresses `dsw-ljvgtooejpvpxmhhc6` in `cn-shanghai` with profile
  `open-dreamer`.
- It rejects any GPU/spec other than the registered NVIDIA A10
  `ecs.gn7i-c8g1.2xlarge`.
- A `Failed` or `Stopped` cycle sends exactly one `start-instance` request.
- `Starting` and `ResourceAllocating` never cause another start request.
- A Running instance must pass the frozen commit, clean worktree, runner hash,
  manifest, checkpoint, Python runtime, GPU-idle and other-pipeline checks.
- The CR-DYN-0008/0009 PID is written with a temporary file plus `mv`.
- A new dynamics launch receives a 12-hour shutdown timer. An already-running
  pipeline is extended to eight hours only when less than four hours remain.
- A stale PID, failed run, dirty worktree, missing input or busy GPU blocks the
  poller; it never repairs, deletes, resets or silently resumes evidence.

The installed LaunchAgent is:

```text
~/Library/LaunchAgents/ai.open-dreamer.dsw-dynamics-poller.plist
```

Inspect its state and append-only operational events:

```bash
launchctl print gui/$(id -u)/ai.open-dreamer.dsw-dynamics-poller
tail -f /Users/xilanhua/WorkSpace/mizzen/mizzen-insight/tmp_agent_output/20260728-open-dreamer-overnight/dsw-dynamics-poller/events.jsonl
```

Disable it without deleting evidence:

```bash
launchctl bootout gui/$(id -u) \
  ~/Library/LaunchAgents/ai.open-dreamer.dsw-dynamics-poller.plist
```

The hourly Codex heartbeat is monitor-only while this LaunchAgent is installed.
It must not issue instance-start or pipeline-launch requests.

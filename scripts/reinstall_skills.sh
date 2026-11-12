#!/usr/bin/env bash
# Reinstall every skill in skills/ into OpenClaw, restart the gateway, and list what is installed.
# Run from the repo root:  bash scripts/reinstall_skills.sh
set -e

# openclaw needs an absolute path; a relative one fails with "Invalid skill slug".
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

for dir in "$ROOT"/skills/*/; do
  name="$(basename "$dir")"
  echo "Installing skill: $name"
  openclaw skills install "$ROOT/skills/$name" --global --force
done

echo "Restarting gateway"
launchctl kickstart -k "gui/$UID/ai.openclaw.gateway"

openclaw skills list

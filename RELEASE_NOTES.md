# UniRoboSim Genesis 0.1.0

Initial standalone release of the local Genesis adapter.

- UniRoboSim v0alpha6 provider, compatible with UniRoboSim >=0.10.9,<0.11.
- Uses unmodified official Genesis World 1.4.2; no custom engine fork.
- State/control, planning geometry, contacts, lifecycle/reset, explicit USD asset profiles,
  render-state application, and headless camera integration.
- Optional engine dependencies remain isolated from adapter import.

Validation on 2026-10-02:

- 60 non-engine/non-GPU tests passed.
- Official Genesis 1.4.2 CPU environment-selection/persistent-wrench smoke passed.
- Source distribution and wheel built successfully.

This release validation does not claim a new full GPU suite or production Mission run.
See README.md for supported asset profiles, camera limitations and explicit unsupported features.
The older docs/VALIDATION.md is historical experimental evidence only.

GitHub release assets include a wheel, source distribution and SHA256SUMS.
No PyPI publication is implied.

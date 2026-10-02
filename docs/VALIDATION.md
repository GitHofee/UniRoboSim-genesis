# Historical validation ledger

> This document records historical experimental SDK checks from 2026-09-29.
> It is not the current supported dependency or release acceptance. The released
> adapter uses unmodified official `genesis-world==1.4.2`; see README.md and
> RELEASE_NOTES.md. Experimental SDK/fork results below do not validate that route.

Historical engine: adapter 0.1.0 with the then-installed local SDK
`genesis-world==1.4.2+fastsim.4` and trimesh 4.12.2, 2026-09-29.
These checks do not claim a completed production Mission or finalized recording.
Evidence lives in `/home/ubuntu/workspace/jdy/platform/integration/20260929-genesis-backend`.

The installed SDK4 complete adapter suite passed on CPU: 23 tests in 59.12 s
(`adapter-tests-fastsim4-cpu.log`), and CUDA: 23 tests in 45.04 s
(`adapter-tests-fastsim4-cuda.log`). These include the new independent
multi-environment physical weld and exhaustive instance-proxy provenance tests.
The unchanged eight-entity SDK4 scene built and stepped three times successfully:
5,893 native geoms, 628,974,267 packed SDF cells, 2,953 unique grids, 1,863.58 s
cold build (`production-build-fastsim4-result.json`). This is a build diagnostic,
not official Mission/recording acceptance. The retained table below was collected
with SDK3; those historical counts are not SDK4 results.

| Check | Result | Evidence |
| --- | --- | --- |
| Complete adapter suite, CPU | 21 passed, 10.72 s | `adapter-tests-fastsim3-cpu-final.log` |
| Complete adapter suite, CUDA | 21 passed, 49.22 s | `adapter-tests-fastsim3-cuda-final.log` |
| Selected upstream USD parser regressions | 10 passed, 26 deliberately deselected, 46.36 s | `sdk-upstream-usd-regressions-fastsim3-final.log` |
| Focused SDK importer tests before wheel freeze | 12 passed, 50.39 s | `sdk-joint-reference-tests-final.log` |
| G2 native CUDA build | 65 links, 43 DOFs, 8 CONNECT closures, 78 frames; 14.83 s | `robot-build-fastsim3-ndarray-result.json` |
| G2 initial absolute positions vs configured values | maximum error 7.73e-8 rad or m | Same G2 result |
| G2 closed-loop anchor residual after three steps | maximum 2.21e-6 m | Same G2 result |
| Independent original Isaac tick0 comparison | 77 frames within 3.934e-6 m and 6.644e-5 degrees | Root-owned `compare_initial_poses.py` and comparison result |
| Remaining base_link vs authored USD composition | 9.58e-8 m; rotation matrix max error 8.33e-9 | `genesis-base-link-authored-parity.json` |

The original recording's nonphysical `base_link` is an outlier by about 24 m.
It is reported separately; the adapter agrees with the source USD composed into
the configured WorldSpec pose and does not reproduce the old reference anomaly.

The eight real-engine adapter cases exercise rigid and articulation reads/control,
nonzero USD joint references and offset joint anchors, named-frame kinematics,
planning geometry leases and lifetime, persistent wrenches, batch selection,
partial reset, stale handles, uint32 seed handling, physical weld hold/release,
duplicate command behavior, capacity rejection without moving the child,
wrapped physical root versus asset-root poses and twists, authored self-collision
policy, and cross-entity native contacts. Unit checks cover optional-import safety,
configuration, exact asset binding and preserved USD collision authoring.

The production scene probe and official FastSim CLI generation are separate
acceptance gates. Camera rendering has an implementation but is not accepted by
this ledger. No Mission or cuRobo implementation was changed.

SDK4 scopes duplicate weld detection to selected environments. The adapter now
requires that exact engine and removes its previous explicit rejection of a shared
pair in distinct environments. The regression attaches the same pair in both
environments, releases one, verifies physical hold versus free fall, reattaches,
and partially resets the other environment. This passed through the real CPU and
CUDA adapter with the installed SDK4 wheel.

SDK4 source validation before wheel installation: three USD per-prim cooking
regressions passed (9.11 s); 22 authored/upstream parser cases passed (135.95 s,
26 deliberately deselected); three SDF query/contact cases passed on CPU (28.26 s)
and CUDA (75.54 s); two native environment-scoped weld cases passed (9.14 s).
These are local regression checks, not original production 1a acceptance.

## Worker-thread lifecycle correction

The first official CLI attempt stopped at the Record output-directory permission
guard, before running Mission. Root separately repaired those directory permissions.
During shutdown, the adapter left its worker-created CUDA runtime for main-thread
atexit, reproducing `CUDA_ERROR_INVALID_CONTEXT` with a single native box and
exit 134 (`lifecycle-diagnosis/before-cuda.log`). The adapter now reference-counts
native sessions and destroys only its own final runtime on the native owner thread.

A second native stack (`lifecycle-diagnosis/gdb-cuda-exit-deep.log`) located an
exiting PyTorch thread-local DeviceContext destructor acquiring the GIL during
Python finalization. Genesis init changes the default Torch device/dtype; its
destroy does not restore them. The adapter preserves the original context's
absence versus presence and restores only its own changed defaults. External
runtimes retain caller ownership; the test's external caller performs its own
final native shutdown and Torch context cleanup.

Five focused CUDA subprocess lifecycle cases passed in 26.88 s; all five controlled
native worker scripts also exited 0 after owner-thread Torch context restoration.
The final complete adapter suite, including a sixth prior-defaults subprocess case,
passed on CPU (29 tests, 74.17 s) and CUDA (29 tests, 123.56 s). Logs are
`lifecycle-diagnosis/adapter-all-cpu-final.log` and `adapter-all-cuda-final.log`.
The subprocess tests use production `QD_KERNEL_COVERAGE=0`; the optional Quadrants
pytest coverage plugin adds a separate reset/atexit hook. Its behavior is not used
to explain the independently reproduced generic PyTorch TLS shutdown abort.

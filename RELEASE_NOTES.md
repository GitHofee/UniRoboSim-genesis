# UniRoboSim Genesis 0.1.1

Requires UniRoboSim >=0.10.10,<0.11 for the appearance, deformable topology and particle-color contracts. Uses unmodified official Genesis World 1.4.2; no custom engine fork.

- Capture effective native material and directional-light appearance for FSR restoration, including stable URDF visual bindings, base-color PNG resources, UV coordinates and topology checksums.
- Read fixed-count SPH state and static per-particle linear RGBA while preserving original indices across native color groups. Uniform color retains the single-entity path; palette size is bounded explicitly.
- Read actual native volumetric FEM topology and state. Remeshing requires explicit preserve_total_mass policy plus Young modulus and Poisson ratio. Unsupported pinning, force controls, material graphs and ambiguous bindings fail explicitly.
- Preserve source input color semantics through linear-to-sRGB adaptation; capture records effective rasterizer BRDF values rather than ignored intended values.

Validation on 2026-10-03:

- 76 CPU tests passed, 30 native-engine tests deselected.
- Official Genesis GPU mixed-scene capture: 121 frames, 601 physics ticks, 132 native FEM nodes, 344 tetrahedra, no inverted tetrahedra; rigid drop, grounded joint rotation and 216 three-color fluid particles exercised together.
- Actual PNG/UV capture smoke passed. One hundred repeated captures were identical, averaging 0.221 ms in the small fixture.
- Source appearance is captured automatically; restoration is supported by the corresponding Isaac backend. Genesis live apply_appearance and soft render-state replay remain explicitly unsupported.

Appearance capture currently covers procedural bodies, supported FEM/SPH and one-to-one URDF visuals. USD/MJCF appearance bindings and arbitrary shader graphs are not advertised. Historical GPU/profiling evidence is not a claim of general engine parity or production Mission validation.

GitHub release assets include a wheel, source distribution and SHA256SUMS. No PyPI publication is implied.

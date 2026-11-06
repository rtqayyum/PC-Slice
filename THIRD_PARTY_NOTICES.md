# Third-party notices

- SurRoL: https://github.com/med-air/SurRoL, pinned revision `a903fa6de40b41e1cad12ddd2d4a6758ec78e817`. The source tool acquires upstream assets separately. Retain upstream LICENSE and instrument-asset provenance, including the upstream robot README.
- Network parameter source: https://github.com/UVA-DSA/telesurgery-qos-analysis, pinned revision `420e659edd79a0a2bf8755fa424888ad3a555b7e`. Acquired files retain their upstream license. `src/network_replay.py` implements source-parameterized stochastic emulation, not bit-identical source execution or a measured network trace.
- Qwen, Mistral, and Phi checkpoints: official repositories and immutable revisions are in `study/checkpoints.json`. Weights are excluded and retain their publishers' licenses/access conditions.
- Python, PyBullet, numerical libraries, PyTorch/Transformers, LM Format Enforcer, TikZ/PGFPlots, and Tectonic retain their respective licenses. Dependencies are installed separately.
- The included observations are authored simulation/engineering-case outputs; no patient data or recorded surgeon requests are included.

The original project-code license is pending author selection. Public visibility is not a replacement for a redistribution license; this snapshot does not relicense upstream assets or weights.

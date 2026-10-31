# Attribution and release scope

- **FastWAM:** https://github.com/hollowyyy/fastwam — upstream backbone and original policy implementation. This portfolio is an experimental extension, not the official FastWAM repository.
- **LIBERO:** https://github.com/Lifelong-Robot-Learning/LIBERO — benchmark and simulator assets, not redistributed here.
- **LIBERO-Plus:** https://github.com/sylvestf/LIBERO-plus — perturbation benchmark, not redistributed here.
- **PyTorch:** https://pytorch.org/ — optional dependency for the MLP component.
- **Matplotlib:** https://matplotlib.org/ — optional dependency for figure reproduction.

The portfolio ships experiment-specific routing components and curated aggregate results. Upstream source trees, model weights, benchmark assets, private infrastructure settings and raw observations are excluded.

The Task4 MLP and scheduling components were extracted from the experiment workspace. Their origin and adjustments are documented in code and in the result provenance records. The 40-task historical runner is not included; its aggregate results must not be attributed to a new run of the Task4 code.

No project-wide license has been selected for this release candidate. Upstream projects and separately obtained checkpoints retain their own licenses. Public hosting alone is not a substitute for choosing a license or permission to redistribute upstream materials.

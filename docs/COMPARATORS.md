# Literature and implemented comparisons

| Implemented method | Role | Relationship to prior art |
| --- | --- | --- |
| Fixed short / long / extended | Fixed horizon and refresh controls | Share finite-plan transport, feedback, checker, fallback |
| Event plan | Conventional adaptive refresh | Inspired by event-triggered scheduling; not a reproduction of Tabuada's theorem/controller |
| ET-MPC* / periodic MPC* | Reduced nonlinear shooting adaptations | Three piecewise IK gain multipliers; SLSQP eight iterations; related to Puttschneider et al. below |
| PC-Slice joint plan | Proposed horizon/refresh selection | Same task and link; no allocation-superiority result established |
| Joint/fixed short | Horizon-selection ablation | Keeps joint refresh and fixes the short horizon |
| Joint without checker | Receiver-checker ablation | Same task/transport; checking removed |

Puttschneider et al., “Towards Event-Triggered NMPC for Efficient 6G Communications: Experimental Results and Open Problems,” ICCA 2025 author preprint: https://arxiv.org/abs/2409.18589v2 . Its full controller stabilizes a Furuta pendulum over an O-RAN platform. Our three-gain articulated-instrument adaptations are different numerical implementations and do not reproduce that hardware/theorem.

Packetized future-input buffering is established prior art: Quevedo and Nesic, IEEE TAC 2011, https://doi.org/10.1109/TAC.2010.2095950 . A receiver shadow check does not implement the formal guarantees of Ames et al., IEEE TAC 2017, https://doi.org/10.1109/TAC.2016.2638961 , or Wabersich and Zeilinger's predictive safety filters/barriers, https://doi.org/10.1016/j.automatica.2021.109597 and https://doi.org/10.1109/TAC.2022.3175628 .

These state-of-the-art relationships are qualitative scope comparisons. Numerical plots compare the actual implemented methods under matched conditions; they do not establish superiority over faithful reproductions of those published systems.

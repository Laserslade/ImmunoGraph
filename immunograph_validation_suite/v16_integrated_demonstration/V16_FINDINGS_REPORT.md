# V16: Integrated Demonstration Findings

One mechanistic narrative linking dynamical state, network topology, perturbation, and intervention, computed end to end in this run.

## 1-4. Model, sampling, simulation, sensitivity, and graph construction

An 8 node signed tanh network was simulated under a single injury pulse (amplitude 5.0, centered at t=15.0). A 20 run ensemble with jittered weights, decay rates, and injury severity was also generated for the ensemble level stages below.

## 5. Graph state discovery

Clustering into 4 regimes against deviation derived ground truth gave trajectory ARI = 0.752 and graph ARI = 0.845 on the index case.

## 6. Temporal network analysis

The graph distance signal peaked at t=12.0 and the trajectory rate of change signal peaked at t=14.0, against a true injury center of t=15.0.

## 7. Motif and module discovery

- **coherent_ffl**: baseline=1.000, activation=0.600, peak=0.000, recovery=0.976 (kruskal p=0.00000)
- **positive_feedback**: baseline=1.000, activation=0.350, peak=0.000, recovery=0.952 (kruskal p=0.00000)
- **hub_centralization**: baseline=0.126, activation=0.505, peak=0.603, recovery=0.278 (kruskal p=0.00000)

## 8. Perturbation analysis

Across pulse amplitudes 0.5 to 10.0, raw reorganization distance R rose from 1.132 to 2.264, while the reorganization fraction (shape distance over raw distance) stayed within 0.216 to 0.245 across the whole range.

## 9-10. Intervention simulation and network restoration

- **symptomatic**: biomarker restoration = 0.693, network restoration = 0.862
- **targeted**: biomarker restoration = 0.692, network restoration = 0.793
- **mistargeted**: biomarker restoration = 0.681, network restoration = 0.805

## Integrated narrative

A single injury pulse produces a fast, saturating deviation from baseline. The dynamic graph representation recovers this same episode as a sequence of network states, and clustering those states against the true injury timeline succeeds well above chance for both the trajectory and the graph representation on this index case. Across the ensemble, the network's coherent feed forward and positive feedback motifs collapse during the peak response and hub centralization rises at the same time, linking motif-level structure directly to dynamical regime. The perturbation analysis shows this reorganization is dominated by magnitude change rather than architecture change, though a consistent minority of the total distance is genuine reorganization at every severity tested. Finally, the intervention comparison shows that different post injury strategies can reach similar biomarker outcomes while leaving the network in measurably different states, and that in this model network restoration never lags behind biomarker restoration for any strategy tested. Together these stages support treating the dynamic graph as a complementary measurement layer, not a replacement for biomarker trajectories, since each stage surfaced structure that the other representation did not fully capture on its own.
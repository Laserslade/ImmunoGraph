# ImmunoGraph

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/NetworkX-graph%20theory-orange" alt="NetworkX">
  <img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/domain-computational%20neuroimmunology-6f42c1" alt="Domain">
  <img src="https://img.shields.io/badge/method-systems%20biology-critical" alt="Method">
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status">
</p>

A computational framework that converts mechanistic neuroimmune ODE models,
covering cytokines, microglia, astrocytes, and neuronal health, into
dynamic, sensitivity-weighted interaction graphs. Edge weights are derived
directly from local sensitivity analysis of the governing equations rather
than inferred statistically from observed time series, so the resulting
network structure is mechanistically grounded rather than correlational.
This graph representation is then used to track how the dominant
interaction structure of the system shifts with age, to surface recurring
network motifs across a simulated population, and to compare candidate
combination therapies by how much they shift the system's graph state
relative to a baseline trajectory.

Most systems biology pipelines either stop at plotting raw state
trajectories, which makes higher order interaction structure hard to see,
or they infer interaction networks statistically from data using methods
such as Granger causality or transfer entropy. Since the underlying
mechanistic equations here are already known, this project treats network
inference as unnecessary and instead reads the interaction graph directly
off the model's own sensitivity structure at each time point. Chaining
this across simulated age trajectories and parameter samples turns a
single static graph into a dynamic sequence, which is what makes motif
discovery, state clustering, and transition detection possible in the
first place.

## What this project does

- Solves a composite neuroimmune ODE system spanning a validated backbone
  model plus an added TBI cytokine module (IL-1beta, IL-4, IL-12).
- Computes local sensitivity of every state variable with respect to every
  other state variable, at any given simulated age.
- Converts each sensitivity matrix into a directed, signed interaction
  graph, where edge sign indicates whether one variable drives another up
  or suppresses it.
- Builds a full sequence of these graphs across simulated age and across a
  sampled population of parameter perturbations.
- Detects recurring structural motifs and dominant hub modules across the
  resulting graph population using clustering.
- Compares candidate combination therapies by measuring how far they shift
  the system's graph state from an untreated baseline.
- Reads the numerical graph output back into natural language findings,
  including hub identification, module concentration, and longitudinal
  transition detection.

## Project structure

```
ImmunoGraph/
├── config.py                        # Central path configuration
├── setup_environment.py             # Dependency check and folder setup
├── phase1_base_model.py             # Base ODE model, solved and plotted
├── phase2_diversity_check.py        # Parameter sampling diversity check
├── phase2_perturbation_harness.py   # Perturbation sweep over parameter grid
├── phase3_composite_model.py        # Composite backbone plus TBI cytokine module
├── phase4_local_sensitivity.py      # Local sensitivity computation
├── phase5_graph_construction.py     # Single sensitivity graph construction
├── phase6_graph_sequence.py         # Dynamic graph sequence across age
├── phase7_batch_pipeline.py         # Batch graph generation across full sim library
├── phase8_motif_discovery.py        # Motif and dominant module discovery
├── phase8_visualizations.py         # Graph and cluster visualizations
├── phase9_therapy_exploration.py    # Combination therapy comparisons
├── phase10_interpretation_layer.py  # Graph statistics to natural language findings
├── experiment_graph_vs_trajectory.py# Graph representation vs raw trajectory comparison
├── models/                          # ODE model definitions
├── sims/                            # Saved simulation runs and manifest
├── graphs/                          # Saved graph objects, sequences, and manifests
└── results/                         # Figures and the compiled findings report
```

## Requirements

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
networkx>=3.1
pandas>=2.0
scikit-learn>=1.3
```

## Running locally

```bash
git clone https://github.com/Laserslade/ImmunoGraph.git
cd ImmunoGraph
pip install -r requirements.txt
python setup_environment.py
```

Each phase script is standalone and can be run independently once the
environment is set up, for example:

```bash
python phase5_graph_construction.py
python phase8_motif_discovery.py
```

## Design notes

No script hardcodes a personal or environment specific file path. Every
path used across the project is resolved through `config.py`. Each phase
is a self contained script rather than one large pipeline, so intermediate
outputs can be inspected at every stage before moving to the next. The
choice to derive edges from mechanistic sensitivity rather than from
statistical inference is deliberate, since the interaction structure is
already known from the governing equations and re-inferring it from
simulated output would only reintroduce uncertainty that the mechanistic
model does not have.

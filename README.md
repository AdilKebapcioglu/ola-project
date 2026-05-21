# OLA Project — Online Learning for Advertising Campaigns

**Course:** Online Learning Applications (OLA) — M. Castiglioni, Politecnico di Milano  
**Group members:** Adil Kebapcioglu, Omar Azab

---

## Problem statement

An agency runs `N` advertising campaigns across `T` rounds. Each round, a
first-price auction is held per campaign: the advertiser wins campaign `i` if
their bid `b_i` meets or exceeds the highest competing bid `m_i`, paying `b_i`
and earning utility `v_i - b_i`. A shared budget `B_total` caps total spend.
A conflict graph `G = (V, E)` forbids bidding simultaneously on campaigns
connected by an edge (e.g., competing products). The goal is to design online
learning algorithms that minimise cumulative regret under these constraints.

---

## Repository structure

```
ola_project/
├── src/
│   ├── utils/          # Seed manager, regret helpers, plotting, multi-trial runner
│   └── core/           # Environments, agents, auction logic, conflict graph
├── notebooks/
│   ├── req1_single_stochastic.ipynb
│   ├── req2_multi_stochastic.ipynb
│   ├── req3_best_of_both_worlds.ipynb
│   └── req4_nonstationary.ipynb
├── report/
│   ├── slides/         # Presentation slides
│   └── figures/        # Exported plots
├── data/
│   ├── configs/        # Experiment parameter configs (JSON)
│   └── seeds/          # Master seeds for reproducibility
├── tests/              # Sanity checks
├── requirements.txt
└── README.md
```

---

## How to reproduce each figure

<!-- Fill in once notebooks are finalised -->

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the relevant notebook under `notebooks/`. Each notebook is self-contained
   and pins its own random seeds — figures are fully reproducible.

---

## Dependencies

See `requirements.txt`. Python 3.13 recommended.

---

## Reproducibility

Every experiment uses a seeded `np.random.Generator` instantiated via
`src/utils/seed_manager.py`. Master seeds are stored in `data/seeds/`. Each
notebook documents its seed at the top cell.

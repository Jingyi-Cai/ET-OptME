
# ET-OptME

# Improving metabolic engineering design with enzyme-thermo optimization

📄 [Published in *Metabolic Engineering*](https://doi.org/10.1016/j.ymben.2025.05.009)

## Overview

**ET-OptME** is a computational framework that integrates enzyme constraints and thermodynamic feasibility into metabolic engineering design. It incorporates two core algorithms:

- **ET-EComp**: Predicts up- or down-regulation targets by comparing enzyme abundance ranges between reference and overproduction states.
- **ET-ESEOF**: Scans enzyme responses to increasing product flux to identify monotonic trends for regulation.

This approach improves prediction accuracy and precision by minimizing enzyme costs and avoiding thermodynamic bottlenecks. ET-OptME targets whole enzymes or enzyme complexes, overcoming the limitations of classical reaction-centric models.

## Features

- Combines stoichiometric, enzyme, and thermodynamic constraints
- Protein-centric target prediction
- Built-in case studies for *Corynebacterium glutamicum*
- Supports COBRApy-compatible metabolic models

## Solver Requirements

- Tested with **Gurobi 9.5** and **CPLEX 12.10**
- Optimization backends required for LP/MILP solving

## Installation

```bash
git clone https://github.com/Jingyi-Cai/ET-OptME.git
cd ET-OptME
# Set up your Python environment (conda recommended)
pip install -r requirements.txt
```

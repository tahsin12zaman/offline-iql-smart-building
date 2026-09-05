# Offline Reinforcement Learning for Smart-Building Energy Management

This project studies offline reinforcement learning for realistic
smart-building energy control using CityLearn.

The main research question is:

> How effectively can offline reinforcement learning learn
> smart-building energy-control policies from previously logged control
> data, without requiring unrestricted online exploration in the target
> building environment?

The project compares Behavior Cloning (BC), Implicit Q-Learning (IQL),
and Conservative Q-Learning (CQL). The main application environment is
CityLearn. Pendulum and MountainCarContinuous-v0 are retained as
supporting benchmark environments.

## Final CityLearn Setup

-   CityLearn version: 2.5.0
-   Buildings: 3
-   Observation dimension: 52
-   Action dimension: 9
-   Offline trajectories: 30
-   Transitions per trajectory: 719
-   Total offline transitions: 21,570
-   Controlled training/evaluation seeds: 1, 2, 3
-   Main training steps: 20,000

## Environment Setup

Activate the CityLearn virtual environment:

``` bash
source .venv_citylearn/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Expected local CityLearn location:

``` text
external/CityLearn
```

Set the local CityLearn paths:

``` bash
export CITYLEARN_LOCAL_DATASETS_PATH="$PWD/external/CityLearn/data/datasets"
export CITYLEARN_LOCAL_MISC_PATH="$PWD/external/CityLearn/data/misc"
export CITYLEARN_OFFLINE=1
```

## Offline Dataset

The final CityLearn offline dataset is:

``` text
data/citylearn_logged_multi.npz
```

It contains observations `(21570, 52)`, actions `(21570, 9)`, rewards
`(21570, 1)`, terminals `(21570,)`, and 30 trajectories. Each resulting
logged trajectory contains 719 transitions.

## Training

The controlled CityLearn experiments use seeds 1, 2, and 3. The seeded
training script is:

``` text
src_app/train_citylearn_algo_cpu_seeded.py
```

The final trained models are the BC, IQL, and CQL `.d3` files for seeds
1, 2, and 3 under `results/`. Seed 0 models are exploratory and are not
part of the final controlled comparison.

## Evaluation

### Controlled Reward Evaluation

``` bash
python src_app/evaluate_citylearn_models.py --seed 1
python src_app/evaluate_citylearn_models.py --seed 2
python src_app/evaluate_citylearn_models.py --seed 3
```

### Application Metrics

``` bash
python src_app/evaluate_citylearn_application_metrics.py
```

This evaluates cumulative RL reward, net electricity consumption,
electricity cost, carbon emissions, and peak net electricity.

### Time-Series Control Behavior

``` bash
python src_app/evaluate_citylearn_timeseries.py
python src_app/plot_citylearn_net_electricity_timeseries_smoothed.py
```

### Native CityLearn KPIs

``` bash
python src_app/evaluate_citylearn_native_kpis.py
python src_app/summarize_citylearn_native_kpis.py
```

### Thermal Comfort

``` bash
python src_app/plot_citylearn_comfort_comparison.py
```

## Final Result Files

The authoritative controlled CityLearn result table is:

``` text
results/citylearn_final_results.csv
```

Additional final result files are:

``` text
results/citylearn_native_kpis.csv
results/citylearn_native_district_kpi_summary.csv
results/citylearn_comfort_summary.csv
```

Important final figures include:

``` text
results/citylearn_normalized_application_summary.png
results/citylearn_per_seed_application_metrics.png
results/citylearn_net_electricity_timeseries_smoothed.png
results/citylearn_thermal_comfort_comparison.png
```

## Main Findings

Across controlled seeds 1, 2, and 3:

-   BC achieves the best cumulative RL reward.
-   IQL achieves lower mean raw net electricity consumption, electricity
    cost, and carbon emissions than BC.
-   BC retains a slight advantage in peak net electricity.
-   CityLearn-native normalized KPIs show BC and IQL are much closer
    than the raw operational metrics suggest.
-   BC has better district-level thermal comfort overall.
-   IQL improves comfort in Buildings 2 and 3 but causes severe
    hot-discomfort behavior in Building 1.
-   CQL performs substantially worse on most energy and district-level
    comfort metrics.

The results demonstrate that offline-RL controller selection depends on
the operational objective being evaluated. Better RL reward or lower
energy-related quantities do not necessarily imply better thermal
comfort or grid behavior.

## Supporting Benchmarks

Pendulum and MountainCarContinuous-v0 are retained as supporting
experiments for offline-RL robustness and dataset-quality analysis.
Their scripts are under `src/`, while the CityLearn application scripts
are under `src_app/`.

## Deployment Scope

The current project is a realistic simulation-based smart-building
control prototype designed around the real operational constraint that
unrestricted online exploration is undesirable in occupied buildings.

Physical deployment in an operating building would require additional
validation and engineering, including real building-management-system
data, explicit operational and comfort constraints, BMS integration, and
staged field testing.
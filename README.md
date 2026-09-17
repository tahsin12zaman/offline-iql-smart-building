## Practical Smart-Building Application

The project includes a user-facing Streamlit application that integrates
the trained offline-RL controllers with the CityLearn evaluation
environment.

Unlike a static results dashboard, the application can load the saved
BC, IQL, and CQL `.d3` policies and execute fresh CityLearn evaluation
rollouts. This provides a practical interface for examining how the
learned controllers affect building operation without performing online
policy training.

The application provides:

- an overview of the project objectives and experimental configuration;
- live evaluation of BC, IQL, and CQL controllers for controlled seeds;
- electricity, cost, carbon, peak-demand, and reward metrics;
- district-level and building-level electricity profiles;
- controlled cross-algorithm comparison;
- building-level thermal-comfort analysis;
- CityLearn-native normalized KPI comparison;
- operational trade-off and failure-case analysis; and
- downloadable evaluation time-series data.

Run the application from the project root with:

```bash
streamlit run app/app.py
```

The application architecture is:

```text
Streamlit user interface
        |
        v
app/citylearn_service.py
        |
        +--> saved BC / IQL / CQL .d3 policies
        |
        +--> offline CityLearn dataset
        |
        v
fresh CityLearn evaluation rollout
        |
        v
energy / cost / carbon / peak / comfort / KPI analysis
```

The application is a simulation-based controller-evaluation and
decision-support prototype. It is not connected to a physical
building-management system.
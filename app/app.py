from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from citylearn_service import run_live_evaluation

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="Smart-Building Offline RL",
    page_icon="🏢",
    layout="wide",
)

st.title("Smart-Building Offline RL Control System")
st.caption(
    "Controller evaluation, operational comparison, "
    "thermal-comfort analysis, and decision support"
)

def read_csv(name):
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None

application = read_csv("citylearn_application_metrics.csv")
comfort = read_csv("citylearn_comfort_summary.csv")
native = read_csv("citylearn_native_district_kpi_summary.csv")

with st.sidebar:
    st.header("Live Controller")
    algorithm = st.selectbox("Controller", ["IQL", "BC", "CQL"])
    seed = st.selectbox("Controlled seed", [1, 2, 3])
    run = st.button(
        "Run Controller Evaluation",
        type="primary",
        width="stretch",
    )

    st.divider()
    st.markdown("**CityLearn configuration**")
    st.write("3 buildings")
    st.write("52 observation features")
    st.write("9 continuous actions")
    st.write("719-step evaluation horizon")

if "live_result" not in st.session_state:
    st.session_state.live_result = None

tabs = st.tabs([
    "Overview",
    "Live Controller",
    "Controller Comparison",
    "Thermal Comfort",
    "Native CityLearn KPIs",
    "Decision Support",
])

with tabs[0]:
    st.subheader("Application Overview")
    st.write(
        "This application demonstrates an offline reinforcement learning workflow "
        "for smart-building energy management. Previously logged CityLearn control "
        "data is used to train policies offline, and saved controllers can then be "
        "evaluated in fresh simulated building rollouts without online learning."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buildings", "3")
    c2.metric("Offline Transitions", "21,570")
    c3.metric("Observation Features", "52")
    c4.metric("Control Actions", "9")

    st.markdown("### Project objectives and deliverables")
    st.markdown(
        """
- Train and evaluate **BC, IQL, and CQL** controllers from logged smart-building data.
- Compare controllers across **energy, cost, carbon, peak demand, and thermal comfort**.
- Provide a **live user-facing interface** for executing saved policies in CityLearn.
- Expose **failure cases and operational trade-offs**, rather than relying on RL reward alone.
- Preserve a reproducible workflow containing trained models, evaluation results, and analysis.
"""
    )

    st.markdown("### Controlled experimental summary")
    s1, s2, s3 = st.columns(3)
    s1.metric(
        "IQL vs BC — Net Electricity",
        "−5.0%",
        help="Mean controlled-seed application metric.",
    )
    s2.metric(
        "IQL vs BC — Electricity Cost",
        "−4.0%",
        help="Mean controlled-seed application metric.",
    )
    s3.metric(
        "IQL vs BC — Carbon",
        "−2.6%",
        help="Mean controlled-seed application metric.",
    )

    st.info(
        "The energy results do not imply that IQL is universally preferable. "
        "BC produced slightly better peak-demand behavior and lower district-level "
        "thermal discomfort, while IQL exhibited a severe Building 1 comfort failure. "
        "The application is designed to make these trade-offs visible."
    )

    st.markdown("### How to use the application")
    st.markdown(
        """
1. Open **Live Controller**, choose BC, IQL, or CQL and a controlled seed.
2. Click **Run Controller Evaluation** to execute the saved policy for up to 719 CityLearn steps.
3. Inspect operational metrics and district/building electricity profiles.
4. Use **Controller Comparison** for controlled-seed aggregate results.
5. Use **Thermal Comfort** and **Native CityLearn KPIs** to inspect operational trade-offs.
6. Use **Decision Support** for interpretation, limitations, and deployment boundaries.
"""
    )

    st.caption(
        "Scope: simulation-based controller evaluation and decision support. "
        "The prototype is not connected to a physical building-management system."
    )

with tabs[1]:
    st.subheader("Execute a Trained Controller")
    st.write(
        "Load a saved BC, IQL, or CQL policy and execute it in a fresh "
        "CityLearn rollout. No online policy improvement occurs during evaluation."
    )

    if run:
        try:
            with st.spinner(
                f"Executing {algorithm} seed {seed} for up to 719 steps..."
            ):
                st.session_state.live_result = run_live_evaluation(
                    algorithm.lower(), int(seed)
                )
            st.success("Controller evaluation completed.")
        except Exception as exc:
            st.session_state.live_result = None
            st.exception(exc)

    if st.session_state.live_result is None:
        st.info(
            "Select a controller and seed in the sidebar, "
            "then click Run Controller Evaluation."
        )
    else:
        metrics, ts = st.session_state.live_result
        st.markdown(
            f"### {metrics['algorithm']} — Seed {metrics['seed']}"
        )

        cols = st.columns(4)
        cols[0].metric(
            "Net Electricity (episode total)",
            f"{metrics['net_electricity_consumption']:,.2f}",
        )
        cols[1].metric(
            "Electricity Cost (episode total)",
            f"{metrics['electricity_cost']:,.2f}",
        )
        cols[2].metric(
            "Carbon Emissions (episode total)",
            f"{metrics['carbon_emission']:,.2f}",
        )
        cols[3].metric(
            "Peak Net Electricity",
            f"{metrics['peak_net_electricity']:,.3f}",
        )

        cols = st.columns(2)
        cols[0].metric(
            "Cumulative Reward",
            f"{metrics['reward']:,.2f}",
        )
        cols[1].metric(
            "Control Steps",
            f"{metrics['steps']}",
        )

        district = ts[ts["level"] == "district"].copy()
        district["24-step moving average"] = (
            district["net_electricity"]
            .rolling(24, min_periods=1)
            .mean()
        )

        fig = px.line(
            district,
            x="step",
            y=["net_electricity", "24-step moving average"],
            labels={
                "value": "Net electricity",
                "step": "Control step",
            },
            title="District Net-Electricity Profile",
        )
        st.plotly_chart(fig, width="stretch")

        buildings = ts[ts["level"] == "building"].copy()
        buildings["building"] = (
            "Building " + buildings["building"].astype(str)
        )

        fig = px.line(
            buildings,
            x="step",
            y="net_electricity",
            color="building",
            labels={
                "net_electricity": "Net electricity",
                "step": "Control step",
            },
            title="Building-Level Electricity Profile",
        )
        st.plotly_chart(fig, width="stretch")

        st.download_button(
            "Download Live Evaluation CSV",
            ts.to_csv(index=False).encode("utf-8"),
            (
                f"citylearn_{metrics['algorithm'].lower()}_"
                f"seed_{metrics['seed']}_live.csv"
            ),
            "text/csv",
        )

with tabs[2]:
    st.subheader("Controlled Algorithm Comparison")

    if application is None:
        st.warning(
            "results/citylearn_application_metrics.csv was not found."
        )
    else:
        df = application.copy()
        if "seed" in df.columns:
            df = df[df["seed"].isin([1, 2, 3])]

        metrics = {
            "net_electricity_consumption": "Net Electricity",
            "electricity_cost": "Electricity Cost",
            "carbon_emission": "Carbon Emissions",
            "peak_net_electricity": "Peak Demand",
        }

        available = {
            key: label
            for key, label in metrics.items()
            if key in df.columns
        }

        if "algorithm" in df.columns and available:
            means = (
                df.groupby("algorithm")[list(available)]
                .mean()
                .reset_index()
            )

            selected = st.selectbox(
                "Comparison metric",
                list(available),
                format_func=lambda x: available[x],
            )

            fig = px.bar(
                means,
                x="algorithm",
                y=selected,
                title=(
                    f"Mean {available[selected]} — "
                    "Controlled Seeds 1–3"
                ),
                labels={
                    selected: available[selected],
                    "algorithm": "Controller",
                },
            )
            st.plotly_chart(fig, width="stretch")

            st.dataframe(
                means.rename(columns=available),
                width="stretch",
                hide_index=True,
            )
        else:
            st.dataframe(df, width="stretch", hide_index=True)

with tabs[3]:
    st.subheader("Thermal Comfort")
    st.write(
        "Mean discomfort proportion is shown for each building. "
        "Lower values indicate less time outside the comfort range."
    )

    if comfort is None:
        st.warning(
            "results/citylearn_comfort_summary.csv was not found."
        )
    else:
        required = {
            "algorithm", "name", "cost_function", "mean", "std"
        }

        if required.issubset(comfort.columns):
            cdf = comfort[
                comfort["cost_function"] == "discomfort_proportion"
            ].copy()

            cdf["mean_percent"] = cdf["mean"] * 100.0
            cdf["std_percent"] = cdf["std"] * 100.0
            cdf["Building"] = cdf["name"].str.replace(
                "_", " ", regex=False
            )

            # Validated district discomfort summary across controlled seeds 1-3.
            # These are district-level statistics, not variability across buildings.
            district = pd.DataFrame(
                {
                    "algorithm": ["BC", "IQL", "CQL"],
                    "mean_percent": [63.88, 66.44, 75.12],
                    "std_percent": [0.74, 2.86, 4.34],
                    "Building": ["District", "District", "District"],
                    "name": ["District", "District", "District"],
                    "cost_function": [
                        "discomfort_proportion",
                        "discomfort_proportion",
                        "discomfort_proportion",
                    ],
                }
            )

            plot_df = pd.concat(
                [
                    cdf[
                        [
                            "algorithm",
                            "Building",
                            "mean_percent",
                            "std_percent",
                        ]
                    ],
                    district[
                        [
                            "algorithm",
                            "Building",
                            "mean_percent",
                            "std_percent",
                        ]
                    ],
                ],
                ignore_index=True,
            )

            order = [
                "District",
                "Building 1",
                "Building 2",
                "Building 3",
            ]
            plot_df["Building"] = pd.Categorical(
                plot_df["Building"],
                categories=order,
                ordered=True,
            )
            plot_df = plot_df.sort_values("Building")

            fig = px.bar(
                plot_df,
                x="Building",
                y="mean_percent",
                color="algorithm",
                barmode="group",
                error_y="std_percent",
                title=(
                    "Mean Thermal Discomfort — "
                    "Controlled Seeds 1–3"
                ),
                labels={
                    "mean_percent": "Discomfort proportion (%)",
                    "algorithm": "Controller",
                },
            )
            st.plotly_chart(fig, width="stretch")

            table = plot_df.copy()
            table["Mean discomfort (%)"] = (
                table["mean_percent"].round(2)
            )
            table["Std (%)"] = table["std_percent"].round(2)

            st.dataframe(
                table[
                    [
                        "algorithm",
                        "Building",
                        "Mean discomfort (%)",
                        "Std (%)",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

            st.warning(
                "Important failure case: IQL's aggregate energy results "
                "must be interpreted together with its building-level "
                "comfort performance. Building 1 exhibits substantially "
                "higher discomfort under IQL."
            )
        else:
            st.dataframe(
                comfort,
                width="stretch",
                hide_index=True,
            )

with tabs[4]:
    st.subheader("Native CityLearn KPIs")
    st.write(
        "Normalized CityLearn indicators provide an independent "
        "evaluation view. Lower values are better for the KPIs shown."
    )

    if native is None:
        st.warning(
            "results/citylearn_native_district_kpi_summary.csv "
            "was not found."
        )
    else:
        required = {
            "algorithm", "cost_function", "mean", "std"
        }

        if required.issubset(native.columns):
            wanted = {
                "electricity_consumption_total":
                    "Electricity Consumption",
                "cost_total":
                    "Cost",
                "carbon_emissions_total":
                    "Carbon Emissions",
                "all_time_peak_average":
                    "All-Time Peak",
                "daily_peak_average":
                    "Daily Peak",
                "ramping_average":
                    "Ramping",
            }

            kdf = native[
                native["cost_function"].isin(wanted)
            ].copy()

            kdf["KPI"] = kdf["cost_function"].map(wanted)

            selected_label = st.selectbox(
                "Native KPI",
                list(wanted.values()),
            )

            selected = kdf[
                kdf["KPI"] == selected_label
            ].copy()

            fig = px.bar(
                selected,
                x="algorithm",
                y="mean",
                error_y="std",
                title=(
                    f"Native CityLearn KPI — {selected_label}"
                ),
                labels={
                    "mean": "Normalized KPI value",
                    "algorithm": "Controller",
                },
            )
            st.plotly_chart(fig, width="stretch")

            pivot_mean = kdf.pivot(
                index="algorithm",
                columns="KPI",
                values="mean",
            ).reset_index()

            preferred = [
                "algorithm",
                "Electricity Consumption",
                "Cost",
                "Carbon Emissions",
                "All-Time Peak",
                "Daily Peak",
                "Ramping",
            ]
            preferred = [
                c for c in preferred
                if c in pivot_mean.columns
            ]

            st.dataframe(
                pivot_mean[preferred].round(4),
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "These are normalized CityLearn KPIs. "
                "They should not be interpreted as the raw electricity, "
                "currency, or carbon values shown in the application metrics."
            )
        else:
            st.dataframe(
                native,
                width="stretch",
                hide_index=True,
            )

with tabs[5]:
    st.subheader(
        "Operational Interpretation and Failure Analysis"
    )

    st.markdown(
        """
### What the experiments demonstrate

**IQL** reduced mean net electricity, operating cost, and carbon
emissions relative to BC in the controlled application metrics.
However, its building-level comfort behavior exposes an important
trade-off, particularly for Building 1.

**BC** provides the behavioral reference. It produced stable results
and the lowest overall district discomfort in the current comfort
analysis, while using somewhat more electricity and producing somewhat
higher cost and carbon than IQL.

**CQL** performed poorly under the current logged dataset and training
configuration, with substantially higher electricity consumption,
cost, carbon emissions, and peak demand.

### Decision-support implication

A controller should not be selected from cumulative RL reward alone.
A building operator should inspect electricity consumption, cost,
carbon emissions, peak demand, normalized CityLearn KPIs, and
building-level thermal comfort together.

The application therefore exposes both successful outcomes and failure
cases instead of treating a single RL score as sufficient evidence of
operational suitability.

### Deployment boundary

This application is a simulation-based smart-building controller
evaluation and decision-support prototype. It executes saved offline-RL
policies in CityLearn. It is not connected to a physical building
management system. Physical deployment would additionally require BMS
integration, explicit safety and comfort constraints, operational
validation, and staged testing.
"""
    )

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
guarded = read_csv("citylearn_guarded_v2_summary.csv")
guard_events = read_csv("citylearn_guarded_v2_events.csv")


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Live Controller")

    algorithm = st.selectbox(
        "Controller",
        ["IQL", "BC", "CQL"],
    )

    seed = st.selectbox(
        "Controlled seed",
        [1, 2, 3],
    )

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


tabs = st.tabs(
    [
        "Overview",
        "Live Controller",
        "Controller Comparison",
        "Thermal Comfort",
        "Native CityLearn KPIs",
        "Comfort Guardrail",
        "Operator Decision Support",
    ]
)


# =====================================================================
# TAB 1 — OVERVIEW
# =====================================================================

with tabs[0]:
    st.subheader("Application Overview")

    st.write(
        "This application demonstrates an offline reinforcement learning "
        "workflow for smart-building energy management. Previously logged "
        "CityLearn control data is used to train policies offline, and "
        "saved controllers can then be evaluated in fresh simulated "
        "building rollouts without online learning."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Buildings",
        "3",
    )

    c2.metric(
        "Offline Transitions",
        "21,570",
    )

    c3.metric(
        "Observation Features",
        "52",
    )

    c4.metric(
        "Control Actions",
        "9",
    )

    st.markdown("### Intended user and operational use case")

    st.write(
        "The intended user is a building energy manager, facilities engineer, "
        "or controls engineer who needs to evaluate candidate smart-building "
        "control policies before considering physical deployment."
    )

    st.markdown(
        """
**Operational problem:** Building operators must balance multiple objectives,
including electricity consumption, operating cost, carbon emissions, peak
demand, and occupant thermal comfort. A controller that performs well on one
objective may perform poorly on another.

**How this application supports the user:** The prototype allows an operator
to execute previously trained offline controllers in a realistic CityLearn
simulation, compare them with a conventional rule-based baseline, inspect
building-level and district-level outcomes, identify failure cases, and
generate decision-support evidence.

**Intended decision:** The application supports pre-deployment screening of
candidate controllers. It helps identify controllers that warrant further
engineering validation and controllers whose observed trade-offs or failure
modes require investigation before any physical-building trial.

**Application boundary:** The system provides simulation-based engineering
decision support. It does not autonomously approve a controller for physical
deployment or directly command a building-management system.
"""
    )

    st.markdown("### Project objectives and deliverables")

    st.markdown(
        """
- Train and evaluate **BC, IQL, and CQL** controllers from logged smart-building data.
- Compare learned controllers with a **conventional rule-based controller (BasicRBC)**.
- Compare controllers across **energy, cost, carbon, peak demand, and thermal comfort**.
- Provide a **live user-facing interface** for executing saved policies in CityLearn.
- Expose **failure cases and operational trade-offs**, rather than relying on RL reward alone.
- Preserve a reproducible workflow containing trained models, evaluation results, and analysis.
"""
    )

    st.markdown("### Engineering controller hierarchy")

    st.write(
        "The evaluation distinguishes three controller categories: "
        "conventional rule-based control (BasicRBC), offline imitation "
        "learning (BC), and offline reinforcement learning (IQL and CQL). "
        "This provides an engineering reference beyond comparisons "
        "between learned algorithms alone."
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
        "The energy results do not imply that IQL is universally "
        "preferable. BC produced slightly better peak-demand behavior "
        "and lower district-level thermal discomfort, while IQL "
        "exhibited a severe Building 1 comfort failure. The application "
        "is designed to make these trade-offs visible."
    )

    st.markdown("### How to use the application")

    st.markdown(
        """
1. Open **Live Controller**, choose BC, IQL, or CQL and a controlled seed.
2. Click **Run Controller Evaluation** to execute the saved policy for up to 719 CityLearn steps.
3. Inspect operational metrics and district/building electricity profiles.
4. Use **Controller Comparison** to compare conventional rule-based control with learned controllers.
5. Use **Thermal Comfort** and **Native CityLearn KPIs** to inspect operational trade-offs.
6. Use **Decision Support** for interpretation, limitations, failure analysis, and deployment boundaries.
"""
    )

    st.caption(
        "Scope: simulation-based controller evaluation and decision "
        "support. The prototype is not connected to a physical "
        "building-management system."
    )


# =====================================================================
# TAB 2 — LIVE CONTROLLER
# =====================================================================

with tabs[1]:
    st.subheader("Execute a Trained Controller")

    st.write(
        "Load a saved BC, IQL, or CQL policy and execute it in a fresh "
        "CityLearn rollout. No online policy improvement occurs during "
        "evaluation."
    )

    if run:
        try:
            with st.spinner(
                f"Executing {algorithm} seed {seed} "
                "for up to 719 steps..."
            ):
                st.session_state.live_result = run_live_evaluation(
                    algorithm.lower(),
                    int(seed),
                )

            st.success(
                "Controller evaluation completed."
            )

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
            f"### {metrics['algorithm']} — "
            f"Seed {metrics['seed']}"
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

        district = ts[
            ts["level"] == "district"
        ].copy()

        district["24-step moving average"] = (
            district["net_electricity"]
            .rolling(
                24,
                min_periods=1,
            )
            .mean()
        )

        fig = px.line(
            district,
            x="step",
            y=[
                "net_electricity",
                "24-step moving average",
            ],
            labels={
                "value": "Net electricity",
                "step": "Control step",
            },
            title="District Net-Electricity Profile",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        buildings = ts[
            ts["level"] == "building"
        ].copy()

        buildings["building"] = (
            "Building "
            + buildings["building"].astype(str)
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

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.download_button(
            "Download Live Evaluation CSV",
            ts.to_csv(
                index=False
            ).encode("utf-8"),
            (
                f"citylearn_"
                f"{metrics['algorithm'].lower()}_"
                f"seed_{metrics['seed']}_live.csv"
            ),
            "text/csv",
        )


# =====================================================================
# TAB 3 — CONTROLLER COMPARISON
# =====================================================================

with tabs[2]:
    st.subheader("Controller Comparison")

    st.write(
        "Compare the learned controllers with a conventional CityLearn "
        "rule-based controller. BC, IQL, and CQL values are means across "
        "controlled seeds 1–3. BasicRBC is a fixed deterministic "
        "rule-based baseline evaluated over the same 719-step scenario."
    )

    if application is None:
        st.warning(
            "results/citylearn_application_metrics.csv "
            "was not found."
        )

    else:
        df = application.copy()

        if "seed" in df.columns:
            df = df[
                df["seed"].isin(
                    [1, 2, 3]
                )
            ]

        metrics = {
            "net_electricity_consumption":
                "Net Electricity",

            "electricity_cost":
                "Electricity Cost",

            "carbon_emission":
                "Carbon Emissions",

            "peak_net_electricity":
                "Peak Building Electricity",
        }

        available = {
            key: label
            for key, label in metrics.items()
            if key in df.columns
        }

        if (
            "algorithm" in df.columns
            and available
        ):
            means = (
                df.groupby(
                    "algorithm"
                )[list(available)]
                .mean()
                .reset_index()
            )

            # ---------------------------------------------------------
            # Conventional rule-based baseline
            #
            # BasicRBC was evaluated using:
            # - the same CityLearn schema,
            # - centralized control,
            # - the same 719-step horizon,
            # - the same metric definitions used by the existing
            #   BC/IQL/CQL application evaluator.
            #
            # Unlike BC/IQL/CQL, BasicRBC is deterministic and is not
            # a trained multi-seed controller.
            # ---------------------------------------------------------

            rbc = pd.DataFrame(
                [
                    {
                        "algorithm": "BasicRBC",
                        "net_electricity_consumption":
                            4799.772914,

                        "electricity_cost":
                            151.899818,

                        "carbon_emission":
                            2192.932854,

                        "peak_net_electricity":
                            10.981712,
                    }
                ]
            )

            comparison = pd.concat(
                [
                    rbc,
                    means,
                ],
                ignore_index=True,
            )

            comparison["algorithm"] = (
                comparison["algorithm"]
                .replace(
                    {
                        "bc": "BC",
                        "iql": "IQL",
                        "cql": "CQL",
                    }
                )
            )

            controller_order = [
                "BasicRBC",
                "BC",
                "IQL",
                "CQL",
            ]

            comparison["algorithm"] = (
                pd.Categorical(
                    comparison["algorithm"],
                    categories=controller_order,
                    ordered=True,
                )
            )

            comparison = (
                comparison
                .sort_values(
                    "algorithm"
                )
                .reset_index(
                    drop=True
                )
            )

            selected = st.selectbox(
                "Comparison metric",
                list(available),
                format_func=lambda x:
                    available[x],
            )

            fig = px.bar(
                comparison,
                x="algorithm",
                y=selected,
                title=(
                    f"{available[selected]} — "
                    "Controller Comparison"
                ),
                labels={
                    selected:
                        available[selected],

                    "algorithm":
                        "Controller",
                },
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            display = (
                comparison.copy()
            )

            display[
                "Controller Type"
            ] = (
                display["algorithm"]
                .astype(str)
                .map(
                    {
                        "BasicRBC":
                            "Conventional rule-based",

                        "BC":
                            "Offline imitation learning",

                        "IQL":
                            "Offline reinforcement learning",

                        "CQL":
                            "Offline reinforcement learning",
                    }
                )
            )

            display = (
                display.rename(
                    columns=available
                )
            )

            display = display.rename(columns={"algorithm": "Controller"})

            columns = [
                "Controller",
                "Controller Type",
                *available.values(),
            ]

            st.dataframe(
                display[
                    columns
                ].round(3),
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "BasicRBC is one deterministic conventional-control "
                "evaluation. BC, IQL, and CQL are mean results across "
                "controlled training seeds 1–3; therefore no seed "
                "variability is attributed to BasicRBC."
            )

            st.info(
                "BasicRBC provides a conventional engineering "
                "reference. Its inclusion distinguishes the comparison "
                "between rule-based control, offline imitation learning, "
                "and offline reinforcement learning."
            )

        else:
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
            )


# =====================================================================
# TAB 4 — THERMAL COMFORT
# =====================================================================

with tabs[3]:
    st.subheader("Thermal Comfort")

    st.write(
        "Mean discomfort proportion is shown for each building. "
        "Lower values indicate less time outside the comfort range."
    )

    if comfort is None:
        st.warning(
            "results/citylearn_comfort_summary.csv "
            "was not found."
        )

    else:
        required = {
            "algorithm",
            "name",
            "cost_function",
            "mean",
            "std",
        }

        if required.issubset(
            comfort.columns
        ):
            cdf = comfort[
                comfort["cost_function"]
                == "discomfort_proportion"
            ].copy()

            cdf["mean_percent"] = (
                cdf["mean"] * 100.0
            )

            cdf["std_percent"] = (
                cdf["std"] * 100.0
            )

            cdf["Building"] = (
                cdf["name"]
                .str.replace(
                    "_",
                    " ",
                    regex=False,
                )
            )

            # Validated district discomfort summary
            # across controlled seeds 1–3.
            #
            # These are district-level statistics,
            # not variability across buildings.

            district = pd.DataFrame(
                {
                    "algorithm": [
                        "BC",
                        "IQL",
                        "CQL",
                    ],

                    "mean_percent": [
                        63.88,
                        66.44,
                        75.12,
                    ],

                    "std_percent": [
                        0.74,
                        2.86,
                        4.34,
                    ],

                    "Building": [
                        "District",
                        "District",
                        "District",
                    ],

                    "name": [
                        "District",
                        "District",
                        "District",
                    ],

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

            plot_df["Building"] = (
                pd.Categorical(
                    plot_df["Building"],
                    categories=order,
                    ordered=True,
                )
            )

            plot_df = (
                plot_df.sort_values(
                    "Building"
                )
            )

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
                    "mean_percent":
                        "Discomfort proportion (%)",

                    "algorithm":
                        "Controller",
                },
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            table = (
                plot_df.copy()
            )

            table[
                "Mean discomfort (%)"
            ] = (
                table[
                    "mean_percent"
                ].round(2)
            )

            table[
                "Std (%)"
            ] = (
                table[
                    "std_percent"
                ].round(2)
            )

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
                "Important failure case: IQL's aggregate energy "
                "results must be interpreted together with its "
                "building-level comfort performance. Building 1 "
                "exhibits substantially higher discomfort under IQL."
            )

        else:
            st.dataframe(
                comfort,
                width="stretch",
                hide_index=True,
            )


# =====================================================================
# TAB 5 — NATIVE CITYLEARN KPIs
# =====================================================================

with tabs[4]:
    st.subheader(
        "Native CityLearn KPIs"
    )

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
            "algorithm",
            "cost_function",
            "mean",
            "std",
        }

        if required.issubset(
            native.columns
        ):
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
                native[
                    "cost_function"
                ].isin(wanted)
            ].copy()

            kdf["KPI"] = (
                kdf[
                    "cost_function"
                ].map(wanted)
            )

            selected_label = (
                st.selectbox(
                    "Native KPI",
                    list(
                        wanted.values()
                    ),
                )
            )

            selected = kdf[
                kdf["KPI"]
                == selected_label
            ].copy()

            fig = px.bar(
                selected,
                x="algorithm",
                y="mean",
                error_y="std",
                title=(
                    "Native CityLearn KPI — "
                    f"{selected_label}"
                ),
                labels={
                    "mean":
                        "Normalized KPI value",

                    "algorithm":
                        "Controller",
                },
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            pivot_mean = (
                kdf.pivot(
                    index="algorithm",
                    columns="KPI",
                    values="mean",
                )
                .reset_index()
            )

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
                c
                for c in preferred
                if c in pivot_mean.columns
            ]

            st.dataframe(
                pivot_mean[
                    preferred
                ].round(4),
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "These are normalized CityLearn KPIs. "
                "They should not be interpreted as the raw "
                "electricity, currency, or carbon values shown "
                "in the application metrics."
            )

        else:
            st.dataframe(
                native,
                width="stretch",
                hide_index=True,
            )


# =====================================================================
# TAB 6 — SUPERVISORY COMFORT GUARDRAIL
# =====================================================================

with tabs[5]:
    st.subheader("Supervisory Comfort Guardrail")

    st.write(
        "This experiment evaluates a simulation-validated supervisory fallback for "
        "IQL Seed 1. At each control step, the latest completed CityLearn thermal "
        "state is checked. If a building is occupied and its indoor temperature "
        "exceeds the cooling setpoint plus the CityLearn comfort band, only that "
        "building's cooling-device action is replaced by the BasicRBC cooling action."
    )

    st.code(
        """IQL proposes 9 centralized actions
        |
        v
Latest completed state: building.time_step - 1
        |
        v
Occupied AND T > cooling setpoint + comfort band?
        |
   +----+----+
   |         |
  No        Yes
   |         |
IQL action   Replace affected building's cooling action
             with BasicRBC cooling
   |         |
   +----+----+
        |
        v
     CityLearn""",
        language="text",
    )

    if guarded is None:
        st.warning("results/citylearn_guarded_v2_summary.csv was not found.")
    else:
        gdf = guarded.copy()
        if "mode" in gdf.columns and {"NORMAL_IQL", "GUARDED_IQL"}.issubset(set(gdf["mode"])):
            normal_row = gdf[gdf["mode"] == "NORMAL_IQL"].iloc[0]
            guarded_row = gdf[gdf["mode"] == "GUARDED_IQL"].iloc[0]

            st.markdown("### IQL Seed 1 — normal vs guarded")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Net Electricity", f"{guarded_row['net_electricity_consumption']:,.2f}", delta=f"{((guarded_row['net_electricity_consumption']/normal_row['net_electricity_consumption'])-1)*100:+.2f}%", delta_color="inverse")
            c2.metric("Electricity Cost", f"{guarded_row['electricity_cost']:,.2f}", delta=f"{((guarded_row['electricity_cost']/normal_row['electricity_cost'])-1)*100:+.2f}%", delta_color="inverse")
            c3.metric("Carbon Emissions", f"{guarded_row['carbon_emission']:,.2f}", delta=f"{((guarded_row['carbon_emission']/normal_row['carbon_emission'])-1)*100:+.2f}%", delta_color="inverse")
            c4.metric("Peak Electricity", f"{guarded_row['peak_net_electricity']:,.3f}", delta=f"{((guarded_row['peak_net_electricity']/normal_row['peak_net_electricity'])-1)*100:+.2f}%", delta_color="inverse")

            comfort_rows = []
            for b in (1, 2, 3):
                key = f"building_{b}_overheating_discomfort"
                comfort_rows += [
                    {"Building": f"Building {b}", "Mode": "Normal IQL", "Overheating discomfort (%)": float(normal_row[key]) * 100.0},
                    {"Building": f"Building {b}", "Mode": "Guarded IQL", "Overheating discomfort (%)": float(guarded_row[key]) * 100.0},
                ]
            guard_comfort = pd.DataFrame(comfort_rows)
            fig = px.bar(guard_comfort, x="Building", y="Overheating discomfort (%)", color="Mode", barmode="group", title="Occupied Overheating Discomfort — IQL Seed 1")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(guard_comfort.round(2), width="stretch", hide_index=True)

            worst_before = float(normal_row["worst_building_overheating_discomfort"]) * 100.0
            worst_after = float(guarded_row["worst_building_overheating_discomfort"]) * 100.0
            activations = int(guarded_row.get("guardrail_activations", 0))
            st.success(
                f"Worst-building overheating discomfort decreased from {worst_before:.2f}% "
                f"to {worst_after:.2f}% in this simulation experiment. The guardrail "
                f"made {activations} cooling-action interventions."
            )
            st.warning(
                "The comfort improvement has a measurable resource penalty: net electricity "
                "increased by 24.79%, electricity cost by 28.23%, carbon emissions by 22.74%, "
                "and peak building electricity by 16.90% for this IQL Seed 1 experiment."
            )

            if guard_events is not None and "building" in guard_events.columns:
                st.markdown("### Intervention evidence")
                counts = guard_events.groupby("building").size().reset_index(name="Interventions")
                counts["Building"] = "Building " + counts["building"].astype(str)
                st.dataframe(counts[["Building", "Interventions"]], width="stretch", hide_index=True)
                with st.expander("Show guardrail intervention records"):
                    st.dataframe(guard_events, width="stretch", hide_index=True)

            st.info(
                "Validation boundary: this is a simulation-validated supervisory comfort "
                "guardrail in CityLearn, not a physically validated building safety system. "
                "Physical deployment would require sensor/BMS integration, actuator and "
                "communication validation, explicit fail-safe behavior, operator override, "
                "and staged commissioning."
            )
        else:
            st.dataframe(gdf, width="stretch", hide_index=True)


# =====================================================================
# TAB 7 — DECISION SUPPORT
# =====================================================================

with tabs[6]:
    st.subheader("Operator Decision Support")

    st.write(
        "Select an operational priority to inspect the measured controller "
        "evidence most relevant to that objective. This panel supports "
        "engineering judgement; it does not treat a single metric as proof "
        "of operational suitability."
    )

    priority = st.selectbox(
        "Operational priority",
        [
            "Reduce Energy Consumption",
            "Reduce Operating Cost",
            "Reduce Carbon Emissions",
            "Limit Peak Demand",
            "Prioritize Thermal Comfort",
        ],
    )

    learned_means = None
    if application is not None and "algorithm" in application.columns:
        decision_df = application.copy()
        if "seed" in decision_df.columns:
            decision_df = decision_df[decision_df["seed"].isin([1, 2, 3])]
        learned_means = (
            decision_df.groupby("algorithm")[
                [
                    "net_electricity_consumption",
                    "electricity_cost",
                    "carbon_emission",
                    "peak_net_electricity",
                ]
            ]
            .mean()
            .reset_index()
        )
        learned_means["algorithm"] = learned_means["algorithm"].str.upper()

    rbc_metrics = {
        "net_electricity_consumption": 4799.772914,
        "electricity_cost": 151.899818,
        "carbon_emission": 2192.932854,
        "peak_net_electricity": 10.981712,
    }

    priority_map = {
        "Reduce Energy Consumption": (
            "net_electricity_consumption",
            "Net Electricity",
        ),
        "Reduce Operating Cost": (
            "electricity_cost",
            "Electricity Cost",
        ),
        "Reduce Carbon Emissions": (
            "carbon_emission",
            "Carbon Emissions",
        ),
        "Limit Peak Demand": (
            "peak_net_electricity",
            "Peak Building Electricity",
        ),
    }

    st.markdown("### Evidence for the selected priority")

    report_lines = [
        "SMART-BUILDING OFFLINE RL — OPERATOR SUMMARY",
        "=" * 52,
        f"Selected operational priority: {priority}",
        "Evaluation horizon: 719 CityLearn control steps",
        "Buildings: 3",
        "",
    ]

    if priority in priority_map and learned_means is not None:
        metric_key, metric_label = priority_map[priority]

        evidence = learned_means[["algorithm", metric_key]].copy()
        evidence = evidence.rename(
            columns={
                "algorithm": "Controller",
                metric_key: metric_label,
            }
        )

        evidence = pd.concat(
            [
                pd.DataFrame(
                    {
                        "Controller": ["BasicRBC"],
                        metric_label: [rbc_metrics[metric_key]],
                    }
                ),
                evidence,
            ],
            ignore_index=True,
        )

        order = ["BasicRBC", "BC", "IQL", "CQL"]
        evidence["Controller"] = pd.Categorical(
            evidence["Controller"],
            categories=order,
            ordered=True,
        )
        evidence = evidence.sort_values("Controller").reset_index(drop=True)

        fig = px.bar(
            evidence,
            x="Controller",
            y=metric_label,
            title=f"Operator Evidence — {metric_label}",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(evidence.round(3), width="stretch", hide_index=True)

        lowest_row = evidence.loc[evidence[metric_label].astype(float).idxmin()]
        st.info(
            f"For this measured metric, {lowest_row['Controller']} recorded "
            f"the lowest value ({float(lowest_row[metric_label]):,.3f}). "
            "This is metric-specific evidence, not a general controller "
            "recommendation. Thermal comfort and other operational outcomes "
            "must also be considered."
        )

        if priority in {
            "Reduce Energy Consumption",
            "Reduce Operating Cost",
            "Reduce Carbon Emissions",
        }:
            st.warning(
                "IQL performs strongly on the aggregate energy/cost/carbon "
                "metrics, but its Building 1 thermal-discomfort result is a "
                "major operational constraint. Do not interpret aggregate "
                "efficiency alone as evidence of safe deployment."
            )

        report_lines += [
            f"Measured evidence: {metric_label}",
        ]
        for _, row in evidence.iterrows():
            report_lines.append(
                f"  {row['Controller']}: {float(row[metric_label]):.3f}"
            )
        report_lines += [
            "",
            "Interpretation:",
            "The lowest value above is specific to the selected metric and",
            "does not constitute an overall controller recommendation.",
        ]

    elif priority == "Prioritize Thermal Comfort":
        if comfort is None:
            st.warning("Thermal-comfort results were not found.")
            report_lines.append("Thermal-comfort results were unavailable.")
        else:
            comfort_decision = comfort[
                comfort["cost_function"] == "discomfort_proportion"
            ].copy()
            comfort_decision["Mean discomfort (%)"] = (
                comfort_decision["mean"] * 100.0
            )
            comfort_decision["Std (%)"] = comfort_decision["std"] * 100.0
            comfort_decision["Building"] = comfort_decision["name"].str.replace(
                "_", " ", regex=False
            )

            comfort_table = comfort_decision[
                [
                    "algorithm",
                    "Building",
                    "Mean discomfort (%)",
                    "Std (%)",
                ]
            ].rename(columns={"algorithm": "Controller"})

            fig = px.bar(
                comfort_table,
                x="Building",
                y="Mean discomfort (%)",
                color="Controller",
                barmode="group",
                error_y="Std (%)",
                title="Building-Level Thermal-Comfort Evidence",
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                comfort_table.round(2),
                width="stretch",
                hide_index=True,
            )

            st.warning(
                "Comfort performance is heterogeneous across buildings. "
                "IQL shows approximately 97.80% mean discomfort in Building 1, "
                "while its Building 2 and Building 3 discomfort is lower. "
                "CQL shows a different failure pattern, with high discomfort "
                "in Buildings 2 and 3. BC is more consistent across the three "
                "buildings. BasicRBC is excluded because equivalent RBC "
                "comfort statistics have not been evaluated."
            )

            report_lines += [
                "Thermal-comfort evidence (mean discomfort %):",
            ]
            for _, row in comfort_table.iterrows():
                report_lines.append(
                    f"  {row['Controller']} — {row['Building']}: "
                    f"{float(row['Mean discomfort (%)']):.2f}%"
                )
            report_lines += [
                "",
                "BasicRBC is excluded from the comfort comparison because",
                "equivalent RBC comfort statistics have not been evaluated.",
            ]

    st.markdown("### Engineering acceptance screening")

    st.write(
        "Define scenario-specific acceptance limits and screen the measured "
        "controller results against them. These limits are operator-defined "
        "engineering requirements for this analysis; they are not universal "
        "building-safety thresholds or evidence of physical-deployment approval."
    )

    if learned_means is None:
        st.warning(
            "Application metrics are unavailable, so controller acceptance "
            "screening cannot be performed."
        )
    else:
        # Use BC as a transparent starting reference for the editable scenario
        # limits. These defaults are comparative benchmarks, not safety limits.
        bc_reference = learned_means[
            learned_means["algorithm"] == "BC"
        ]

        if not bc_reference.empty:
            bc_reference = bc_reference.iloc[0]

            default_energy = float(
                bc_reference["net_electricity_consumption"]
            )
            default_cost = float(
                bc_reference["electricity_cost"]
            )
            default_carbon = float(
                bc_reference["carbon_emission"]
            )
            default_peak = float(
                bc_reference["peak_net_electricity"]
            )
        else:
            default_energy = 1500.0
            default_cost = 45.0
            default_carbon = 720.0
            default_peak = 10.0

        default_comfort = 70.0
        worst_comfort = None

        if comfort is not None and {
            "algorithm", "cost_function", "mean"
        }.issubset(comfort.columns):
            acceptance_comfort = comfort[
                comfort["cost_function"] == "discomfort_proportion"
            ].copy()

            if not acceptance_comfort.empty:
                acceptance_comfort["mean_percent"] = (
                    acceptance_comfort["mean"] * 100.0
                )
                worst_comfort = (
                    acceptance_comfort.groupby("algorithm")["mean_percent"]
                    .max()
                    .reset_index()
                )
                worst_comfort["algorithm"] = (
                    worst_comfort["algorithm"].str.upper()
                )

                bc_comfort = worst_comfort[
                    worst_comfort["algorithm"] == "BC"
                ]
                if not bc_comfort.empty:
                    default_comfort = float(
                        bc_comfort.iloc[0]["mean_percent"]
                    )

        st.caption(
            "Initial limits use BC's measured mean performance as a comparative "
            "reference where available. The operator can change every limit. "
            "For comfort, the criterion uses the worst measured building-level "
            "mean discomfort for each learned controller."
        )

        a1, a2, a3 = st.columns(3)
        a4, a5, a6 = st.columns(3)

        energy_limit = a1.number_input(
            "Maximum net electricity",
            min_value=0.0,
            value=float(default_energy),
            step=10.0,
            format="%.3f",
            help="Scenario-specific maximum episode-total net electricity.",
        )
        cost_limit = a2.number_input(
            "Maximum electricity cost",
            min_value=0.0,
            value=float(default_cost),
            step=1.0,
            format="%.3f",
            help="Scenario-specific maximum episode-total electricity cost.",
        )
        carbon_limit = a3.number_input(
            "Maximum carbon emissions",
            min_value=0.0,
            value=float(default_carbon),
            step=10.0,
            format="%.3f",
            help="Scenario-specific maximum episode-total carbon emissions.",
        )
        peak_limit = a4.number_input(
            "Maximum peak building electricity",
            min_value=0.0,
            value=float(default_peak),
            step=0.1,
            format="%.3f",
            help="Maximum observed building-level instantaneous net electricity.",
        )
        comfort_limit = a5.number_input(
            "Maximum building discomfort (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_comfort),
            step=1.0,
            format="%.2f",
            help=(
                "Maximum allowed mean discomfort for the worst measured building "
                "under a learned controller. BasicRBC comfort is not evaluated."
            ),
        )
        warning_band = a6.number_input(
            "Warning band above limit (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
            help=(
                "Values above a limit but within this percentage are marked WARN; "
                "larger exceedances are marked FAIL."
            ),
        )

        acceptance = learned_means.copy()
        acceptance = acceptance.rename(columns={"algorithm": "Controller"})

        acceptance = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "Controller": "BasicRBC",
                            **rbc_metrics,
                        }
                    ]
                ),
                acceptance,
            ],
            ignore_index=True,
        )

        acceptance["Controller"] = acceptance["Controller"].replace(
            {"bc": "BC", "iql": "IQL", "cql": "CQL"}
        )

        if worst_comfort is not None:
            comfort_lookup = dict(
                zip(
                    worst_comfort["algorithm"],
                    worst_comfort["mean_percent"],
                )
            )
        else:
            comfort_lookup = {}

        acceptance["Worst building discomfort (%)"] = acceptance[
            "Controller"
        ].map(comfort_lookup)

        def screening_status(value, limit):
            if pd.isna(value):
                return "NOT EVALUATED"
            value = float(value)
            limit = float(limit)
            warn_limit = limit * (1.0 + float(warning_band) / 100.0)
            if value <= limit:
                return "PASS"
            if value <= warn_limit:
                return "WARN"
            return "FAIL"

        acceptance["Energy"] = acceptance[
            "net_electricity_consumption"
        ].apply(lambda x: screening_status(x, energy_limit))
        acceptance["Cost"] = acceptance[
            "electricity_cost"
        ].apply(lambda x: screening_status(x, cost_limit))
        acceptance["Carbon"] = acceptance[
            "carbon_emission"
        ].apply(lambda x: screening_status(x, carbon_limit))
        acceptance["Peak"] = acceptance[
            "peak_net_electricity"
        ].apply(lambda x: screening_status(x, peak_limit))
        acceptance["Comfort"] = acceptance[
            "Worst building discomfort (%)"
        ].apply(lambda x: screening_status(x, comfort_limit))

        status_columns = ["Energy", "Cost", "Carbon", "Peak", "Comfort"]

        def overall_screen(row):
            statuses = [row[c] for c in status_columns]
            evaluated = [s for s in statuses if s != "NOT EVALUATED"]
            if "FAIL" in evaluated:
                return "FAIL"
            if "WARN" in evaluated:
                return "WARN"
            if evaluated and all(s == "PASS" for s in evaluated):
                return "PASS"
            return "NOT EVALUATED"

        acceptance["Overall screening"] = acceptance.apply(
            overall_screen,
            axis=1,
        )

        controller_order = ["BasicRBC", "BC", "IQL", "CQL"]
        acceptance["Controller"] = pd.Categorical(
            acceptance["Controller"],
            categories=controller_order,
            ordered=True,
        )
        acceptance = acceptance.sort_values("Controller").reset_index(drop=True)

        screening_table = acceptance[
            [
                "Controller",
                "Energy",
                "Cost",
                "Carbon",
                "Peak",
                "Comfort",
                "Overall screening",
            ]
        ].copy()

        st.dataframe(
            screening_table,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "PASS = measured value is at or below the selected limit. "
            "WARN = measured value exceeds the limit but remains within the "
            "selected warning band. FAIL = measured value exceeds the warning "
            "band. NOT EVALUATED means equivalent evidence is unavailable."
        )

        detail_table = acceptance[
            [
                "Controller",
                "net_electricity_consumption",
                "electricity_cost",
                "carbon_emission",
                "peak_net_electricity",
                "Worst building discomfort (%)",
            ]
        ].rename(
            columns={
                "net_electricity_consumption": "Net Electricity",
                "electricity_cost": "Electricity Cost",
                "carbon_emission": "Carbon Emissions",
                "peak_net_electricity": "Peak Building Electricity",
            }
        )

        with st.expander("Show measured values used for screening"):
            st.dataframe(
                detail_table.round(3),
                width="stretch",
                hide_index=True,
            )

        st.info(
            "The overall screening result is deliberately conservative: any "
            "FAIL produces an overall FAIL, and any WARN produces an overall "
            "WARN unless another evaluated criterion fails. Missing BasicRBC "
            "comfort evidence is shown as NOT EVALUATED rather than being "
            "treated as either success or failure."
        )

        report_lines += [
            "",
            "ENGINEERING ACCEPTANCE SCREENING",
            (
                "Limits: net electricity <= "
                f"{energy_limit:.3f}; cost <= {cost_limit:.3f}; "
                f"carbon <= {carbon_limit:.3f}; peak <= {peak_limit:.3f}; "
                f"worst-building discomfort <= {comfort_limit:.2f}%"
            ),
            f"Warning band above each limit: {warning_band:.1f}%",
        ]

        for _, row in screening_table.iterrows():
            report_lines.append(
                f"  {row['Controller']}: Energy={row['Energy']}, "
                f"Cost={row['Cost']}, Carbon={row['Carbon']}, "
                f"Peak={row['Peak']}, Comfort={row['Comfort']}, "
                f"Overall={row['Overall screening']}"
            )

        report_lines += [
            "These are operator-defined scenario criteria, not universal safety",
            "limits or authorization for physical-building deployment.",
        ]

    st.markdown("### Reproducibility specification")

    st.write(
        "The application is backed by a reproducible experimental workflow that "
        "separates offline policy development, saved controller artifacts, fresh "
        "evaluation, engineering analysis, and the user-facing decision-support layer."
    )

    reproducibility = pd.DataFrame(
        [
            {
                "Stage": "1. Logged dataset",
                "Specification": "CityLearn smart-building control data",
                "Reproducibility evidence": "21,570 offline transitions used for policy development.",
            },
            {
                "Stage": "2. Offline policy development",
                "Specification": "BC, IQL, and CQL",
                "Reproducibility evidence": "Policies are trained from logged transitions without online environment learning.",
            },
            {
                "Stage": "3. Controlled repetitions",
                "Specification": "Seeds 1, 2, and 3",
                "Reproducibility evidence": "Learned-controller comparison and comfort summaries use the controlled seed set.",
            },
            {
                "Stage": "4. Saved controller artifacts",
                "Specification": "Persisted .d3 policy models",
                "Reproducibility evidence": "Live Controller loads the selected saved algorithm/seed policy rather than retraining it.",
            },
            {
                "Stage": "5. Evaluation environment",
                "Specification": "CityLearn; 3 buildings; 52 observations; 9 continuous actions",
                "Reproducibility evidence": "Centralized controller evaluation uses a fixed 719-step horizon.",
            },
            {
                "Stage": "6. Conventional reference",
                "Specification": "CityLearn BasicRBC",
                "Reproducibility evidence": "Evaluated on the same schema, horizon, and application-metric definitions.",
            },
            {
                "Stage": "7. Evaluation outputs",
                "Specification": "Energy, cost, carbon, peak, comfort, and native CityLearn KPIs",
                "Reproducibility evidence": "Results are preserved as analysis CSV artifacts and exposed by the application.",
            },
            {
                "Stage": "8. Application layer",
                "Specification": "Streamlit decision-support interface",
                "Reproducibility evidence": "Saved-policy execution, controller comparison, acceptance screening, failure analysis, and downloadable outputs.",
            },
        ]
    )

    st.dataframe(
        reproducibility,
        width="stretch",
        hide_index=True,
    )

    st.code(
        """Logged CityLearn transitions (21,570)
        |
        v
Offline training: BC / IQL / CQL
Controlled seeds: 1 / 2 / 3
        |
        v
Saved .d3 controller artifacts
        |
        v
Fresh CityLearn evaluation
3 buildings | 52 observations | 9 actions | 719 steps
        |
        +----------------------+
        |                      |
        v                      v
Application metrics      Comfort / native KPIs
        |                      |
        +----------+-----------+
                   |
                   v
Streamlit controller evaluation + decision support""",
        language="text",
    )

    st.markdown("**Result artifacts used by the application**")
    st.code(
        """results/citylearn_application_metrics.csv
results/citylearn_comfort_summary.csv
results/citylearn_native_district_kpi_summary.csv
results/citylearn_guarded_v2_summary.csv
results/citylearn_guarded_v2_events.csv
results/citylearn_<algorithm>_seed_<n>.d3""",
        language="text",
    )

    st.caption(
        "Reproducibility scope: the fixed dataset, controlled seed set, saved "
        "controller artifacts, evaluation configuration, and preserved result "
        "tables provide traceable evidence from offline policy development to "
        "the application. Exact package versions are controlled separately by "
        "the repository dependency specification."
    )

    report_lines += [
        "",
        "REPRODUCIBILITY SPECIFICATION",
        "Logged CityLearn transitions: 21,570",
        "Learned controllers: BC, IQL, CQL",
        "Controlled seeds: 1, 2, 3",
        "Environment: 3 buildings, 52 observations, 9 continuous actions",
        "Evaluation horizon: 719 control steps",
        "Saved learned policies: persisted .d3 controller artifacts",
        "Conventional reference: CityLearn BasicRBC",
        "Application result artifacts:",
        "  results/citylearn_application_metrics.csv",
        "  results/citylearn_comfort_summary.csv",
        "  results/citylearn_native_district_kpi_summary.csv",
        "  results/citylearn_guarded_v2_summary.csv",
        "  results/citylearn_guarded_v2_events.csv",
        "The repository dependency specification controls package versions.",
    ]

    st.markdown("### Engineering requirements and verification")

    requirements = pd.DataFrame(
        [
            {
                "ID": "R1",
                "Engineering requirement": "Use logged building-control data for offline policy development.",
                "Implementation": "21,570 logged CityLearn transitions; BC/IQL/CQL trained offline.",
                "Verification evidence": "Dataset, training pipeline, and saved controller models.",
            },
            {
                "ID": "R2",
                "Engineering requirement": "Execute trained controllers without online policy improvement.",
                "Implementation": "Live Controller loads a saved policy and performs a fresh CityLearn rollout.",
                "Verification evidence": "719-step live evaluation with downloadable time-series output.",
            },
            {
                "ID": "R3",
                "Engineering requirement": "Compare learned control against a conventional reference.",
                "Implementation": "CityLearn BasicRBC conventional rule-based baseline.",
                "Verification evidence": "Same schema, horizon, and application-metric definitions.",
            },
            {
                "ID": "R4",
                "Engineering requirement": "Quantify operational outcomes beyond RL reward.",
                "Implementation": "Energy, cost, carbon, peak, native KPIs, and thermal comfort.",
                "Verification evidence": "Controller Comparison, Thermal Comfort, and Native KPI tabs.",
            },
            {
                "ID": "R5",
                "Engineering requirement": "Expose operational trade-offs and controller failure cases.",
                "Implementation": "Building-level comfort analysis and operator-priority evidence panel.",
                "Verification evidence": "IQL Building 1 and CQL building-level comfort failure patterns are surfaced.",
            },
            {
                "ID": "R6",
                "Engineering requirement": "Provide a practically demonstrable user-facing prototype.",
                "Implementation": "Interactive Streamlit application with controller execution and decision support.",
                "Verification evidence": "Local/public application workflow and downloadable evaluation outputs.",
            },
            {
                "ID": "R7",
                "Engineering requirement": "Mitigate detected occupied overheating while preserving traceable supervisory behavior.",
                "Implementation": "Simulation-validated comfort guardrail replaces only an affected building's IQL cooling-device action with BasicRBC cooling.",
                "Verification evidence": "IQL Seed 1 normal-vs-guarded evaluation, 207 intervention records, and quantified comfort/resource trade-offs.",
            },
        ]
    )

    st.dataframe(requirements, width="stretch", hide_index=True)

    st.markdown("### Failure and safety analysis")

    failures = pd.DataFrame(
        [
            {
                "Failure mode": "IQL Building 1 thermal discomfort",
                "Observed evidence": "Mean discomfort approximately 97.80% across controlled seeds.",
                "Operational consequence": "Aggregate energy improvement may coexist with unacceptable local comfort.",
                "Required mitigation before physical deployment": "Explicit comfort constraints, monitoring, validation, and conservative fallback control.",
            },
            {
                "Failure mode": "CQL unstable operational performance",
                "Observed evidence": "High electricity, cost, carbon, peak demand, and high discomfort in Buildings 2–3.",
                "Operational consequence": "Current CQL configuration is not supported as operationally suitable by these experiments.",
                "Required mitigation before physical deployment": "Revisit dataset coverage, training/configuration, constraint handling, and revalidate before consideration.",
            },
            {
                "Failure mode": "Single-metric controller selection",
                "Observed evidence": "Controller ordering changes across energy, peak, and building-level comfort measures.",
                "Operational consequence": "Optimizing one KPI can conceal adverse outcomes elsewhere.",
                "Required mitigation before physical deployment": "Use multi-KPI acceptance criteria and building-level constraint checks.",
            },
            {
                "Failure mode": "Simulation-to-building transfer",
                "Observed evidence": "Current prototype executes in CityLearn and is not connected to a physical BMS.",
                "Operational consequence": "Simulation results do not establish physical-building safety or performance.",
                "Required mitigation before physical deployment": "BMS integration, staged testing, hardware/communication validation, operator override, and monitored commissioning.",
            },
        ]
    )

    st.dataframe(failures, width="stretch", hide_index=True)

    st.markdown("### System architecture and data flow")

    st.code(
        """CityLearn Building Environment (3 buildings, 52 observations, 9 actions)
        |
        v
Logged Operational Transitions (21,570)
        |
        +-------------------------------+
        |                               |
        v                               v
Offline Learning Pipeline         Conventional Baseline
BC / IQL / CQL                    BasicRBC
        |                               |
        v                               v
Saved Learned Policies            Fixed Rule-Based Policy
        |                               |
        +---------------+---------------+
                        |
                        v
              Fresh 719-Step Evaluation
                        |
                        v
       Operational + Comfort + Native KPI Analysis
                        |
                        v
          Operator Decision-Support Application
                        |
                        v
       Evidence, Failure Warnings, Downloadable Report""",
        language="text",
    )

    st.caption(
        "The Streamlit interface is the presentation and decision-support "
        "layer of the workflow; policy development, saved models, evaluation, "
        "and KPI analysis remain separate engineering components."
    )

    st.markdown("### Deployment boundary")
    st.write(
        "This is a simulation-based smart-building controller evaluation "
        "and decision-support prototype. It is not connected to a physical "
        "building-management system. Physical deployment would additionally "
        "require BMS integration, explicit safety and comfort constraints, "
        "operational validation, hardware and communication interfaces, "
        "operator override/fallback mechanisms, and staged commissioning."
    )

    report_lines += [
        "",
        "ENGINEERING BOUNDARY",
        "This report summarizes simulation evidence only. The prototype is",
        "not connected to a physical building-management system. Physical",
        "deployment requires explicit constraints, BMS integration, validation,",
        "fallback/override mechanisms, and staged commissioning.",
    ]

    report_text = "\n".join(report_lines)

    st.markdown("### Download operator summary")
    st.download_button(
        "Download Operator Summary",
        report_text.encode("utf-8"),
        "smart_building_operator_summary.txt",
        "text/plain",
    )


from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

import pandas as pd
import plotly.express as px
import streamlit as st

from citylearn_service import run_live_evaluation


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _git_value(*args):
    """Return repository metadata without making report generation depend on Git."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unavailable"


def _create_evidence_identity():
    generated = datetime.now(timezone.utc)
    revision = _git_value("rev-parse", "--short", "HEAD")
    branch = _git_value("branch", "--show-current") or "detached"

    timestamp_id = generated.strftime("%Y%m%dT%H%M%SZ")
    evidence_id = f"CL-EVID-{timestamp_id}-{revision}"

    return {
        "id": evidence_id,
        "generated_utc": generated.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "revision": revision,
        "branch": branch,
    }


if "evidence_identity" not in st.session_state:
    st.session_state.evidence_identity = _create_evidence_identity()

evidence_identity = st.session_state.evidence_identity


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


def read_json(name):
    path = RESULTS / name
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


experiment_manifest = read_json("experiment_manifest.json")

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
        "Engineering Scenario",
        "Defense Demo",
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

        # -------------------------------------------------------------
        # Consolidated controller evaluation report
        # -------------------------------------------------------------
        st.markdown("### Controller evaluation report")

        st.write(
            "Generate a consolidated engineering evidence package containing "
            "the evaluated controller results, operator-defined acceptance "
            "criteria, screening outcomes, reproducibility information, and "
            "validation limitations."
        )

        evaluation_report_lines = list(report_lines)

        evaluation_report_lines += [
            "",
            "=" * 72,
            "CONTROLLER EVALUATION REPORT",
            "=" * 72,
            "",
            "EVIDENCE IDENTITY",
            f"Evidence ID: {evidence_identity['id']}",
            f"Generated: {evidence_identity['generated_utc']}",
            f"Repository revision: {evidence_identity['revision']}",
            f"Repository branch: {evidence_identity['branch']}",
            "Evidence scope: CityLearn simulation/application verification",
            "Primary offline dataset: data/citylearn_logged_multi.npz",
            "",
            "1. PURPOSE",
            (
                "Engineering decision-support report for comparing offline "
                "smart-building controllers under a common CityLearn "
                "evaluation configuration."
            ),
            (
                "The report combines measured controller performance with "
                "operator-defined engineering acceptance screening."
            ),
            "",
            "2. EXPERIMENTAL CONFIGURATION",
            "Environment: CityLearn",
            "Buildings: 3",
            "Observation dimension: 52",
            "Continuous action dimension: 9",
            "Evaluation horizon: 719 control steps",
            "Offline dataset: 21,570 logged transitions",
            "Learned controllers: BC, IQL, CQL",
            "Controlled training seeds: 1, 2, 3",
            "Conventional reference controller: CityLearn BasicRBC",
            "",
            "3. MEASURED CONTROLLER RESULTS",
        ]

        for _, row in acceptance.iterrows():
            controller_name = str(row["Controller"])
            discomfort = row["Worst building discomfort (%)"]

            if pd.isna(discomfort):
                discomfort_text = "NOT EVALUATED"
            else:
                discomfort_text = f"{float(discomfort):.2f}%"

            evaluation_report_lines += [
                f"Controller: {controller_name}",
                (
                    "  Net electricity: "
                    f"{float(row['net_electricity_consumption']):.3f}"
                ),
                (
                    "  Electricity cost: "
                    f"{float(row['electricity_cost']):.3f}"
                ),
                (
                    "  Carbon emissions: "
                    f"{float(row['carbon_emission']):.3f}"
                ),
                (
                    "  Peak building electricity: "
                    f"{float(row['peak_net_electricity']):.3f}"
                ),
                f"  Worst-building discomfort: {discomfort_text}",
                (
                    "  Screening: "
                    f"Energy={row['Energy']}, "
                    f"Cost={row['Cost']}, "
                    f"Carbon={row['Carbon']}, "
                    f"Peak={row['Peak']}, "
                    f"Comfort={row['Comfort']}, "
                    f"Overall={row['Overall screening']}"
                ),
                "",
            ]

        evaluation_report_lines += [
            "4. OPERATOR-DEFINED ENGINEERING REQUIREMENTS",
            f"Maximum net electricity: {energy_limit:.3f}",
            f"Maximum electricity cost: {cost_limit:.3f}",
            f"Maximum carbon emissions: {carbon_limit:.3f}",
            f"Maximum peak building electricity: {peak_limit:.3f}",
            (
                "Maximum worst-building discomfort: "
                f"{comfort_limit:.2f}%"
            ),
            f"Warning band above limits: {warning_band:.1f}%",
            "",
            "5. SCREENING LOGIC",
            (
                "PASS: measured value is at or below the selected "
                "operator-defined limit."
            ),
            (
                "WARN: measured value exceeds the selected limit but remains "
                "within the selected warning band."
            ),
            (
                "FAIL: measured value exceeds the selected warning band."
            ),
            (
                "NOT EVALUATED: equivalent measured evidence is unavailable."
            ),
            (
                "Overall screening is conservative: any FAIL produces FAIL; "
                "otherwise any WARN produces WARN; otherwise evaluated "
                "criteria must all PASS."
            ),
            "",
            "6. REPRODUCIBILITY EVIDENCE",
            "Logged CityLearn transitions: 21,570",
            "Offline policy-development algorithms: BC, IQL, CQL",
            "Controlled seeds: 1, 2, 3",
            "Saved learned-controller artifacts: .d3 policy models",
            "Evaluation environment: CityLearn",
            "Fixed centralized evaluation horizon: 719 control steps",
            (
                "Application metrics: "
                "results/citylearn_application_metrics.csv"
            ),
            (
                "Comfort evidence: "
                "results/citylearn_comfort_summary.csv"
            ),
            (
                "Native CityLearn KPI evidence: "
                "results/citylearn_native_district_kpi_summary.csv"
            ),
            "",
            "7. ENGINEERING INTERPRETATION",
            (
                "A controller that performs strongly on aggregate energy, "
                "cost, or carbon metrics may still fail an operational "
                "requirement such as building-level thermal comfort."
            ),
            (
                "The acceptance screen therefore treats controller selection "
                "as a multi-criterion engineering decision rather than an "
                "optimization of a single aggregate metric."
            ),
            "",
            "8. VALIDATION BOUNDARY",
            (
                "Results in this report are simulation-based engineering "
                "evidence obtained from the specified CityLearn experiment."
            ),
            (
                "Acceptance limits are operator-defined scenario requirements "
                "and are not universal building-safety thresholds."
            ),
            (
                "PASS does not constitute authorization for physical-building "
                "deployment."
            ),
            (
                "Physical deployment would require BMS integration, sensor "
                "and actuator validation, explicit fail-safe behavior, "
                "operator override, communication validation, and staged "
                "commissioning."
            ),
            "",
            "9. TRACEABILITY",
            (
                "Controller evaluation -> measured engineering metrics -> "
                "operator-defined requirements -> PASS/WARN/FAIL screening -> "
                "decision-support evidence."
            ),
            "",
            "END OF CONTROLLER EVALUATION REPORT",
        ]

        evaluation_report_text = "\n".join(
            str(line) for line in evaluation_report_lines
        )

        st.code(
            "\n".join(
                [
                    "Engineering evidence package ready",
                    f"Evidence ID: {evidence_identity['id']}",
                    f"Generated: {evidence_identity['generated_utc']}",
                    f"Repository revision: {evidence_identity['revision']}",
                    f"Controllers evaluated: {len(acceptance)}",
                    "Environment: CityLearn",
                    "Evaluation horizon: 719 control steps",
                    (
                        "Contents: configuration + measured results + "
                        "acceptance screening + reproducibility + limitations"
                    ),
                ]
            ),
            language="text",
        )

        st.download_button(
            "Download Controller Evaluation Report",
            evaluation_report_text.encode("utf-8"),
            "controller_evaluation_report.txt",
            "text/plain",
            key="download_controller_evaluation_report",
        )

        # -------------------------------------------------------------
        # Formal controller verification record
        # -------------------------------------------------------------
        st.markdown("### Controller verification record")

        st.write(
            "Generate a traceable verification record for a selected controller "
            "using the same measured results, operator-defined limits, warning "
            "band, and PASS/WARN/FAIL logic shown above."
        )

        verification_controller = st.selectbox(
            "Controller for verification record",
            ["BasicRBC", "BC", "IQL", "CQL"],
            key="verification_controller",
        )

        verification_row = acceptance[
            acceptance["Controller"].astype(str) == verification_controller
        ]

        if verification_row.empty:
            st.warning(
                "Verification evidence is unavailable for the selected controller."
            )
        else:
            verification_row = verification_row.iloc[0]

            verification_status = str(
                verification_row["Overall screening"]
            )

            if verification_status == "PASS":
                status_statement = "REQUIREMENTS SATISFIED"
            elif verification_status == "WARN":
                status_statement = "REQUIREMENTS REQUIRE OPERATOR REVIEW"
            elif verification_status == "FAIL":
                status_statement = "REQUIREMENTS NOT SATISFIED"
            else:
                status_statement = "VERIFICATION INCOMPLETE"

            verification_lines = [
                "CONTROLLER VERIFICATION RECORD",
                "=" * 72,
                "",
                "1. EVALUATION IDENTITY",
                f"Evidence ID: {evidence_identity['id']}",
                f"Generated: {evidence_identity['generated_utc']}",
                f"Repository revision: {evidence_identity['revision']}",
                f"Repository branch: {evidence_identity['branch']}",
                "Evidence scope: CityLearn simulation/application verification",
                "Primary offline dataset: data/citylearn_logged_multi.npz",
                f"Controller: {verification_controller}",
                "Environment: CityLearn",
                "Buildings: 3",
                "Observation dimension: 52",
                "Continuous action dimension: 9",
                "Evaluation horizon: 719 control steps",
                (
                    "Controller artifact: CityLearn BasicRBC fixed rule-based "
                    "policy"
                    if verification_controller == "BasicRBC"
                    else (
                        "Controller artifact family: "
                        f"results/citylearn_{verification_controller.lower()}_"
                        "seed_<n>.d3"
                    )
                ),
                "",
                "2. OPERATOR-DEFINED ACCEPTANCE CRITERIA",
                f"Maximum net electricity: {energy_limit:.3f}",
                f"Maximum electricity cost: {cost_limit:.3f}",
                f"Maximum carbon emissions: {carbon_limit:.3f}",
                f"Maximum peak building electricity: {peak_limit:.3f}",
                f"Maximum worst-building discomfort: {comfort_limit:.2f}%",
                f"Warning band above limits: {warning_band:.1f}%",
                "",
                "3. MEASURED VERIFICATION EVIDENCE",
                (
                    "Net electricity: "
                    f"{float(verification_row['net_electricity_consumption']):.3f}"
                    f" | Status: {verification_row['Energy']}"
                ),
                (
                    "Electricity cost: "
                    f"{float(verification_row['electricity_cost']):.3f}"
                    f" | Status: {verification_row['Cost']}"
                ),
                (
                    "Carbon emissions: "
                    f"{float(verification_row['carbon_emission']):.3f}"
                    f" | Status: {verification_row['Carbon']}"
                ),
                (
                    "Peak building electricity: "
                    f"{float(verification_row['peak_net_electricity']):.3f}"
                    f" | Status: {verification_row['Peak']}"
                ),
            ]

            discomfort_value = verification_row[
                "Worst building discomfort (%)"
            ]

            if pd.isna(discomfort_value):
                verification_lines.append(
                    "Worst-building discomfort: NOT EVALUATED "
                    f"| Status: {verification_row['Comfort']}"
                )
            else:
                verification_lines.append(
                    "Worst-building discomfort: "
                    f"{float(discomfort_value):.2f}%"
                    f" | Status: {verification_row['Comfort']}"
                )

            verification_lines += [
                "",
                "4. OVERALL VERIFICATION RESULT",
                f"Screening result: {verification_status}",
                f"Decision statement: {status_statement}",
                "",
                "5. STATUS DEFINITIONS",
                (
                    "PASS: measured value is at or below the selected "
                    "operator-defined limit."
                ),
                (
                    "WARN: measured value exceeds the selected limit but remains "
                    "within the selected warning band."
                ),
                (
                    "FAIL: measured value exceeds the selected warning band."
                ),
                (
                    "NOT EVALUATED: equivalent verification evidence is "
                    "unavailable."
                ),
                "",
                "6. EVIDENCE PROVENANCE",
                "Offline dataset: 21,570 logged CityLearn transitions",
                "Learned controllers: BC, IQL, CQL",
                "Controlled training seeds: 1, 2, 3",
                "Conventional reference: CityLearn BasicRBC",
                "Application metrics: results/citylearn_application_metrics.csv",
                "Comfort evidence: results/citylearn_comfort_summary.csv",
                (
                    "Native KPI evidence: "
                    "results/citylearn_native_district_kpi_summary.csv"
                ),
                "",
                "7. VALIDATION BOUNDARY",
                (
                    "This verification record documents simulation-based "
                    "engineering acceptance screening."
                ),
                (
                    "The acceptance limits are operator-defined scenario "
                    "requirements, not universal building-safety thresholds."
                ),
                (
                    "This record does not authorize physical-building deployment."
                ),
                (
                    "Physical deployment requires BMS integration, explicit "
                    "safety and comfort constraints, operational validation, "
                    "fallback/override mechanisms, and staged commissioning."
                ),
            ]

            verification_text = "\n".join(verification_lines)

            st.code(
                "\n".join(
                    [
                        f"Evidence ID: {evidence_identity['id']}",
                        f"Generated: {evidence_identity['generated_utc']}",
                        f"Repository revision: {evidence_identity['revision']}",
                        f"Controller: {verification_controller}",
                        f"Overall screening: {verification_status}",
                        f"Decision statement: {status_statement}",
                    ]
                ),
                language="text",
            )

            st.download_button(
                "Download Controller Verification Record",
                verification_text.encode("utf-8"),
                (
                    "controller_verification_"
                    f"{verification_controller.lower()}.txt"
                ),
                "text/plain",
                key="download_controller_verification_record",
            )

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

    st.markdown("#### Experiment manifest")

    if experiment_manifest is not None:
        manifest_scope = experiment_manifest["experiment_scope"]
        manifest_artifacts = experiment_manifest["controller_artifacts"]
        manifest_evidence = experiment_manifest["preserved_evidence"]

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Manifest version",
            str(experiment_manifest["manifest_version"]),
        )
        m2.metric(
            "Saved controllers",
            str(
                manifest_artifacts[
                    "available_saved_controller_count"
                ]
            ),
        )
        m3.metric(
            "Controlled artifacts",
            str(
                manifest_artifacts[
                    "controlled_evaluation_artifact_count"
                ]
            ),
        )
        m4.metric(
            "Hashed evidence files",
            str(len(manifest_evidence)),
        )

        st.write(
            "Machine-readable experiment identity: "
            f"{manifest_scope['environment']}; "
            f"{manifest_scope['buildings']} buildings; "
            f"{manifest_scope['observation_features']} observation features; "
            f"{manifest_scope['continuous_actions']} continuous actions; "
            f"{manifest_scope['evaluation_horizon_steps']} control steps; "
            f"{manifest_scope['offline_logged_transitions']:,} logged transitions."
        )

        st.caption(
            "Controlled evaluation seeds: "
            + ", ".join(
                str(seed)
                for seed in manifest_scope[
                    "controlled_evaluation_seeds"
                ]
            )
            + ". SHA-256 hashes preserve the identity of controller "
            "artifacts, engineering evidence files, and verification tests."
        )

        st.download_button(
            "Download Experiment Manifest (JSON)",
            json.dumps(
                experiment_manifest,
                indent=2,
            ).encode("utf-8"),
            "experiment_manifest.json",
            "application/json",
            key="download_experiment_manifest",
        )

        with st.expander("Inspect manifest provenance"):
            st.code(
                "\n".join(
                    [
                        (
                            "Schema: "
                            + experiment_manifest["manifest_schema"]
                        ),
                        (
                            "Manifest version: "
                            + str(
                                experiment_manifest[
                                    "manifest_version"
                                ]
                            )
                        ),
                        (
                            "Dependency specification: "
                            + experiment_manifest[
                                "reproducibility"
                            ]["dependency_specification"]
                        ),
                        (
                            "Manifest generator: "
                            + experiment_manifest[
                                "reproducibility"
                            ]["manifest_generator"]
                        ),
                        (
                            "Current repository revision: "
                            + evidence_identity["revision"]
                        ),
                    ]
                ),
                language="text",
            )

        st.warning(
            experiment_manifest["validation_boundary"]
        )
    else:
        st.warning(
            "Experiment manifest is unavailable. Regenerate it with "
            "scripts/generate_experiment_manifest.py before producing "
            "a reproducibility evidence package."
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
        "  results/experiment_manifest.json",
        "Experiment manifest: machine-readable artifact identity with SHA-256 hashes.",
        "The repository dependency specification controls package versions.",
    ]

    st.markdown("### Engineering requirements and verification")

    requirements = pd.DataFrame(
        [
            {
                "ID": "R1",
                "Engineering requirement": "Use logged building-control data for offline policy development.",
                "Implementation": "21,570 logged CityLearn transitions; BC, IQL, and CQL trained offline.",
                "Verification method": "Inspect dataset dimensions, training configuration, and persisted controller artifacts.",
                "Evidence / artifact": "data/citylearn_logged_multi.npz + results/citylearn_<algorithm>_seed_<n>.d3",
                "Status": "VERIFIED",
            },
            {
                "ID": "R2",
                "Engineering requirement": "Execute trained controllers without online policy improvement.",
                "Implementation": "Live Controller loads a selected saved policy and performs a fresh CityLearn rollout.",
                "Verification method": "Execute saved-policy evaluation and confirm the fixed rollout horizon without retraining.",
                "Evidence / artifact": "Live Controller evaluation + downloadable 719-step time-series output",
                "Status": "VERIFIED",
            },
            {
                "ID": "R3",
                "Engineering requirement": "Compare learned control against a conventional reference.",
                "Implementation": "CityLearn BasicRBC is retained as the conventional rule-based baseline.",
                "Verification method": "Compare controllers under the common CityLearn schema, horizon, and metric definitions.",
                "Evidence / artifact": "results/citylearn_application_metrics.csv + Controller Comparison",
                "Status": "VERIFIED",
            },
            {
                "ID": "R4",
                "Engineering requirement": "Quantify operational outcomes beyond cumulative RL reward.",
                "Implementation": "Energy, cost, carbon, peak demand, native CityLearn KPIs, and thermal comfort are evaluated.",
                "Verification method": "Inspect preserved metric tables and application visualizations.",
                "Evidence / artifact": "citylearn_application_metrics.csv + citylearn_native_district_kpi_summary.csv + citylearn_comfort_summary.csv",
                "Status": "VERIFIED",
            },
            {
                "ID": "R5",
                "Engineering requirement": "Expose operational trade-offs and controller failure cases.",
                "Implementation": "Building-level comfort analysis, multi-KPI comparison, and operator-priority evidence are provided.",
                "Verification method": "Inspect controller-level and building-level results for conflicting operational outcomes.",
                "Evidence / artifact": "Thermal Comfort + Operator Decision Support + failure and safety analysis",
                "Status": "VERIFIED",
            },
            {
                "ID": "R6",
                "Engineering requirement": "Provide a practically demonstrable user-facing engineering prototype.",
                "Implementation": "Interactive Streamlit application provides controller execution, analysis, decision support, and guided workflow.",
                "Verification method": "Execute the application workflow from controller selection through engineering evidence review.",
                "Evidence / artifact": "Live Controller + Engineering Scenario + downloadable outputs",
                "Status": "VERIFIED",
            },
            {
                "ID": "R7",
                "Engineering requirement": "Mitigate detected occupied overheating while preserving traceable supervisory behavior.",
                "Implementation": "The simulation-validated comfort guardrail selectively substitutes BasicRBC cooling for an affected building.",
                "Verification method": "Compare normal and guarded IQL Seed 1 evaluations and verify intervention-event consistency.",
                "Evidence / artifact": "citylearn_guarded_v2_summary.csv + citylearn_guarded_v2_events.csv + test_guardrail_evidence.py",
                "Status": "VERIFIED",
            },
            {
                "ID": "R8",
                "Engineering requirement": "Screen candidate controllers against explicit scenario-specific engineering criteria.",
                "Implementation": "Operator-defined limits and warning bands produce PASS, WARN, FAIL, or NOT EVALUATED outcomes.",
                "Verification method": "Apply the configured limits to measured controller evidence and inspect the resulting screening table.",
                "Evidence / artifact": "Engineering acceptance screening + Controller Verification Record",
                "Status": "VERIFIED",
            },
            {
                "ID": "R9",
                "Engineering requirement": "Preserve traceable engineering evidence for review and reproducibility.",
                "Implementation": "The application consolidates configuration, measured results, screening outcomes, reproducibility information, and validation limitations.",
                "Verification method": "Generate and inspect the downloadable engineering evidence records.",
                "Evidence / artifact": "Controller Evaluation Report + Controller Verification Record + Operator Summary",
                "Status": "VERIFIED",
            },
        ]
    )

    st.dataframe(
        requirements,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Traceability status refers to verification within the documented "
        "CityLearn simulation and application scope. VERIFIED does not mean "
        "certified, commissioned, or validated for physical-building deployment."
    )

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



# =====================================================================
# TAB 8 — GUIDED ENGINEERING SCENARIO
# =====================================================================

with tabs[7]:
    st.subheader("Guided Engineering Scenario")

    st.write(
        "This workflow demonstrates how a building energy manager, facilities "
        "engineer, or controls engineer can use the application to evaluate a "
        "candidate offline-RL controller before considering further deployment "
        "activities. The scenario links controller execution, multi-KPI "
        "assessment, comfort analysis, engineering acceptance screening, "
        "failure mitigation, and verification evidence."
    )

    st.info(
        "Scenario: assess candidate controllers for a three-building CityLearn "
        "portfolio over the fixed 719-step evaluation horizon while considering "
        "energy, operating cost, carbon emissions, peak demand, and thermal "
        "comfort. The objective is engineering evaluation and decision support, "
        "not automatic physical-building deployment."
    )

    st.markdown("### Engineering workflow")

    scenario_steps = pd.DataFrame(
        [
            {
                "Step": "1",
                "Engineering activity": "Define the operational objective",
                "Application evidence": "Operator Decision Support",
                "Expected outcome": "A scenario-specific operational priority and acceptance criteria.",
            },
            {
                "Step": "2",
                "Engineering activity": "Select a candidate controller",
                "Application evidence": "Live Controller",
                "Expected outcome": "BC, IQL, or CQL and a controlled seed are selected.",
            },
            {
                "Step": "3",
                "Engineering activity": "Execute the saved controller",
                "Application evidence": "Live Controller",
                "Expected outcome": "A fresh 719-step CityLearn rollout is produced without online policy learning.",
            },
            {
                "Step": "4",
                "Engineering activity": "Compare operational performance",
                "Application evidence": "Controller Comparison + Native CityLearn KPIs",
                "Expected outcome": "Energy, cost, carbon, peak-demand, and native KPI evidence is reviewed.",
            },
            {
                "Step": "5",
                "Engineering activity": "Check building-level comfort",
                "Application evidence": "Thermal Comfort",
                "Expected outcome": "Local comfort failures that may be hidden by aggregate metrics are identified.",
            },
            {
                "Step": "6",
                "Engineering activity": "Investigate supervisory mitigation",
                "Application evidence": "Comfort Guardrail",
                "Expected outcome": "Normal and guarded IQL evidence is compared, including intervention records and resource trade-offs.",
            },
            {
                "Step": "7",
                "Engineering activity": "Screen against engineering requirements",
                "Application evidence": "Operator Decision Support",
                "Expected outcome": "PASS, WARN, FAIL, or NOT EVALUATED statuses are produced from operator-defined limits.",
            },
            {
                "Step": "8",
                "Engineering activity": "Preserve verification evidence",
                "Application evidence": "Controller Verification Record + Controller Evaluation Report",
                "Expected outcome": "Traceable evidence is generated for review and engineering documentation.",
            },
        ]
    )

    st.dataframe(
        scenario_steps,
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Demonstration case — IQL Seed 1")

    st.write(
        "A concrete case study is available in the application using the saved "
        "IQL Seed 1 controller. The case demonstrates why controller evaluation "
        "must consider more than aggregate energy performance."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Evaluation horizon",
        "719 steps",
    )
    c2.metric(
        "Buildings",
        "3",
    )
    c3.metric(
        "Offline transitions",
        "21,570",
    )
    c4.metric(
        "Guardrail interventions",
        "207",
    )

    st.markdown("#### Case-study sequence")

    st.write(
        "**1. Candidate evaluation:** run or inspect IQL Seed 1 using the same "
        "CityLearn evaluation configuration used for the other learned controllers."
    )

    st.write(
        "**2. Multi-KPI review:** inspect electricity consumption, operating "
        "cost, carbon emissions, peak demand, and native CityLearn KPIs rather "
        "than relying only on cumulative RL reward."
    )

    st.write(
        "**3. Failure identification:** building-level comfort evidence exposes "
        "a severe Building 1 discomfort result even though aggregate efficiency "
        "metrics are comparatively strong."
    )

    st.write(
        "**4. Engineering response:** the simulation-validated supervisory "
        "comfort guardrail replaces only the affected building's IQL cooling "
        "action with BasicRBC cooling when the occupied overheating condition "
        "is satisfied."
    )

    st.write(
        "**5. Verification:** the guarded evaluation records each intervention "
        "and quantifies the resulting comfort-versus-resource trade-off."
    )

    st.write(
        "**6. Decision support:** operator-defined acceptance criteria and the "
        "verification record make the failure and supporting evidence explicit "
        "instead of presenting the controller as universally suitable."
    )

    # -------------------------------------------------------------
    # Quantitative engineering case-study evidence
    # -------------------------------------------------------------
    st.markdown("### Case-study engineering evidence")

    iql_seed1 = None
    if application is not None:
        iql_rows = application[
            (application["algorithm"].astype(str).str.upper() == "IQL")
            & (application["seed"] == 1)
        ]
        if not iql_rows.empty:
            iql_seed1 = iql_rows.iloc[0]

    iql_b1_comfort = None
    if comfort is not None:
        comfort_rows = comfort[
            (comfort["algorithm"].astype(str).str.upper() == "IQL")
            & (comfort["name"] == "Building_1")
            & (comfort["cost_function"] == "discomfort_proportion")
        ]
        if not comfort_rows.empty:
            iql_b1_comfort = float(comfort_rows.iloc[0]["mean"])

    if iql_seed1 is not None:
        st.markdown("#### Candidate controller — IQL Seed 1")

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Net electricity",
            f"{float(iql_seed1['net_electricity_consumption']):,.2f}",
        )
        q2.metric(
            "Electricity cost",
            f"{float(iql_seed1['electricity_cost']):,.2f}",
        )
        q3.metric(
            "Carbon emissions",
            f"{float(iql_seed1['carbon_emission']):,.2f}",
        )
        q4.metric(
            "Peak electricity",
            f"{float(iql_seed1['peak_net_electricity']):,.3f}",
        )

        if iql_b1_comfort is not None:
            st.error(
                "Detected engineering failure: across the controlled three-seed "
                f"IQL evaluation, Building 1 mean discomfort is "
                f"{100.0 * iql_b1_comfort:.2f}%. This demonstrates why aggregate "
                "resource metrics alone are insufficient for controller selection."
            )

    if guarded is not None:
        normal_rows = guarded[
            guarded["mode"].astype(str).str.upper() == "NORMAL_IQL"
        ]
        guarded_rows = guarded[
            guarded["mode"].astype(str).str.upper() == "GUARDED_IQL"
        ]

        if not normal_rows.empty and not guarded_rows.empty:
            normal = normal_rows.iloc[0]
            mitigated = guarded_rows.iloc[0]

            st.markdown("#### Supervisory mitigation — normal vs guarded IQL")

            comparison = pd.DataFrame(
                [
                    {
                        "Measure": "Net electricity",
                        "Normal IQL": float(normal["net_electricity_consumption"]),
                        "Guarded IQL": float(mitigated["net_electricity_consumption"]),
                    },
                    {
                        "Measure": "Electricity cost",
                        "Normal IQL": float(normal["electricity_cost"]),
                        "Guarded IQL": float(mitigated["electricity_cost"]),
                    },
                    {
                        "Measure": "Carbon emissions",
                        "Normal IQL": float(normal["carbon_emission"]),
                        "Guarded IQL": float(mitigated["carbon_emission"]),
                    },
                    {
                        "Measure": "Peak electricity",
                        "Normal IQL": float(normal["peak_net_electricity"]),
                        "Guarded IQL": float(mitigated["peak_net_electricity"]),
                    },
                    {
                        "Measure": "Worst-building overheating discomfort (%)",
                        "Normal IQL": 100.0 * float(
                            normal["worst_building_overheating_discomfort"]
                        ),
                        "Guarded IQL": 100.0 * float(
                            mitigated["worst_building_overheating_discomfort"]
                        ),
                    },
                ]
            )

            comparison["Change (%)"] = (
                (
                    comparison["Guarded IQL"]
                    - comparison["Normal IQL"]
                )
                / comparison["Normal IQL"]
                * 100.0
            )

            st.dataframe(
                comparison.round(3),
                width="stretch",
                hide_index=True,
            )

            normal_discomfort = float(
                normal["worst_building_overheating_discomfort"]
            )
            guarded_discomfort = float(
                mitigated["worst_building_overheating_discomfort"]
            )

            comfort_reduction = (
                (normal_discomfort - guarded_discomfort)
                / normal_discomfort
                * 100.0
            )

            energy_change = (
                (
                    float(mitigated["net_electricity_consumption"])
                    - float(normal["net_electricity_consumption"])
                )
                / float(normal["net_electricity_consumption"])
                * 100.0
            )

            cost_change = (
                (
                    float(mitigated["electricity_cost"])
                    - float(normal["electricity_cost"])
                )
                / float(normal["electricity_cost"])
                * 100.0
            )

            carbon_change = (
                (
                    float(mitigated["carbon_emission"])
                    - float(normal["carbon_emission"])
                )
                / float(normal["carbon_emission"])
                * 100.0
            )

            peak_change = (
                (
                    float(mitigated["peak_net_electricity"])
                    - float(normal["peak_net_electricity"])
                )
                / float(normal["peak_net_electricity"])
                * 100.0
            )

            intervention_count = int(
                float(mitigated["guardrail_activations"])
            )

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Worst-building discomfort",
                f"{100.0 * guarded_discomfort:.2f}%",
                delta=(
                    f"{-100.0 * (normal_discomfort - guarded_discomfort):.2f} "
                    "percentage points"
                ),
            )

            r2.metric(
                "Relative discomfort reduction",
                f"{comfort_reduction:.1f}%",
            )

            r3.metric(
                "Guardrail interventions",
                f"{intervention_count}",
            )

            st.info(
                "Engineering trade-off: the supervisory guardrail reduces "
                f"worst-building overheating discomfort by {comfort_reduction:.1f}% "
                f"relative to normal IQL, while net electricity increases by "
                f"{energy_change:.1f}%, cost by {cost_change:.1f}%, carbon by "
                f"{carbon_change:.1f}%, and peak electricity by {peak_change:.1f}%. "
                "The result therefore demonstrates an explicit comfort-versus-resource "
                "trade-off rather than a universally superior controller."
            )

            if guard_events is not None:
                if len(guard_events) == intervention_count:
                    st.success(
                        f"Verification check: all {intervention_count} reported "
                        "guardrail interventions have corresponding preserved "
                        "event records."
                    )
                else:
                    st.warning(
                        "Verification check: the reported guardrail activation "
                        f"count ({intervention_count}) does not match the preserved "
                        f"event-record count ({len(guard_events)})."
                    )

    if guarded is not None:
        st.markdown("### Preserved guardrail evidence")

        required_guard_cols = {
            "mode",
            "net_electricity_consumption",
            "electricity_cost",
            "carbon_emission",
            "peak_net_electricity",
        }

        if required_guard_cols.issubset(guarded.columns):
            scenario_guard = guarded.copy()

            display_cols = [
                "mode",
                "net_electricity_consumption",
                "electricity_cost",
                "carbon_emission",
                "peak_net_electricity",
            ]

            optional_cols = [
                "building_1_overheating_discomfort",
                "building_2_overheating_discomfort",
                "building_3_overheating_discomfort",
                "interventions",
            ]

            for col in optional_cols:
                if col in scenario_guard.columns:
                    display_cols.append(col)

            st.dataframe(
                scenario_guard[display_cols].round(3),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption(
                "Guardrail summary evidence is available, but its stored schema "
                "does not contain the expected scenario-display columns."
            )

    st.markdown("### What this scenario demonstrates")

    st.success(
        "The application supports an end-to-end engineering evaluation workflow: "
        "define requirements → execute a saved controller → measure operational "
        "and comfort outcomes → identify failure → inspect a supervisory "
        "mitigation → apply acceptance screening → preserve verification evidence."
    )

    st.warning(
        "Validation boundary: this workflow demonstrates simulation-based "
        "engineering evaluation in CityLearn. It does not establish physical "
        "building safety, certify a controller for deployment, or replace BMS "
        "integration, commissioning, operator override, and field validation."
    )
# =====================================================================
# TAB 9 — DEFENSE DEMO
# =====================================================================

with tabs[8]:
    st.subheader("Defense Demo — Engineering Evaluation Workflow")

    st.write(
        "This guided view demonstrates the complete engineering workflow using "
        "the preserved IQL Seed 1 case study: define the control problem, inspect "
        "a candidate controller, detect an operational failure, evaluate a "
        "supervisory mitigation, and preserve reproducible verification evidence."
    )

    st.info(
        "Intended user: building energy manager, facilities engineer, or controls "
        "engineer evaluating candidate controllers before physical deployment."
    )

    # -------------------------------------------------------------
    # 1. Engineering problem
    # -------------------------------------------------------------
    st.markdown("### 1. Engineering problem and system")

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Buildings", "3")
    d2.metric("Observations", "52")
    d3.metric("Actions", "9")
    d4.metric("Offline transitions", "21,570")
    d5.metric("Evaluation horizon", "719 steps")

    st.write(
        "BC, IQL, and CQL policies are trained from previously logged building "
        "control data. Saved policies are then executed in fresh CityLearn "
        "rollouts without online policy improvement and compared using engineering "
        "outcomes rather than cumulative RL reward alone."
    )

    # -------------------------------------------------------------
    # 2. Candidate controller
    # -------------------------------------------------------------
    st.markdown("### 2. Candidate controller — IQL Seed 1")

    demo_iql = None

    if application is not None:
        demo_iql_rows = application[
            (application["algorithm"].astype(str).str.upper() == "IQL")
            & (application["seed"] == 1)
        ]

        if not demo_iql_rows.empty:
            demo_iql = demo_iql_rows.iloc[0]

    if demo_iql is not None:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Net electricity",
            f"{float(demo_iql['net_electricity_consumption']):,.2f}",
        )
        c2.metric(
            "Electricity cost",
            f"{float(demo_iql['electricity_cost']):,.2f}",
        )
        c3.metric(
            "Carbon emissions",
            f"{float(demo_iql['carbon_emission']):,.2f}",
        )
        c4.metric(
            "Peak electricity",
            f"{float(demo_iql['peak_net_electricity']):,.3f}",
        )
    else:
        st.warning("IQL Seed 1 application evidence is unavailable.")

    # -------------------------------------------------------------
    # 3. Failure detection
    # -------------------------------------------------------------
    st.markdown("### 3. Engineering failure detection")

    demo_iql_b1_comfort = None

    if comfort is not None:
        demo_comfort_rows = comfort[
            (comfort["algorithm"].astype(str).str.upper() == "IQL")
            & (comfort["name"] == "Building_1")
            & (comfort["cost_function"] == "discomfort_proportion")
        ]

        if not demo_comfort_rows.empty:
            demo_iql_b1_comfort = float(
                demo_comfort_rows.iloc[0]["mean"]
            )

    if demo_iql_b1_comfort is not None:
        st.error(
            "Building-level verification exposes a failure hidden by aggregate "
            f"resource metrics: IQL Building 1 mean discomfort is "
            f"{100.0 * demo_iql_b1_comfort:.2f}% across the controlled seed "
            "evaluation."
        )
    else:
        st.warning("Building-level IQL comfort evidence is unavailable.")

    st.write(
        "Engineering interpretation: a controller cannot be assessed using "
        "energy, cost, carbon, peak demand, or RL reward in isolation. "
        "Building-level operational constraints must also be checked."
    )

    # -------------------------------------------------------------
    # 4. Supervisory mitigation and trade-off
    # -------------------------------------------------------------
    st.markdown("### 4. Supervisory mitigation and measured trade-off")

    demo_guardrail_ready = False

    if guarded is not None and "mode" in guarded.columns:
        demo_normal_rows = guarded[
            guarded["mode"].astype(str).str.upper() == "NORMAL_IQL"
        ]
        demo_guarded_rows = guarded[
            guarded["mode"].astype(str).str.upper() == "GUARDED_IQL"
        ]

        if not demo_normal_rows.empty and not demo_guarded_rows.empty:
            demo_guardrail_ready = True
            demo_normal = demo_normal_rows.iloc[0]
            demo_guarded = demo_guarded_rows.iloc[0]

            demo_before = 100.0 * float(
                demo_normal["worst_building_overheating_discomfort"]
            )
            demo_after = 100.0 * float(
                demo_guarded["worst_building_overheating_discomfort"]
            )
            demo_reduction = (
                (demo_before - demo_after) / demo_before * 100.0
            )
            demo_interventions = int(
                float(demo_guarded["guardrail_activations"])
            )

            g1, g2, g3 = st.columns(3)
            g1.metric(
                "Worst discomfort — normal",
                f"{demo_before:.2f}%",
            )
            g2.metric(
                "Worst discomfort — guarded",
                f"{demo_after:.2f}%",
                delta=f"{demo_after - demo_before:+.2f} percentage points",
            )
            g3.metric(
                "Guardrail interventions",
                str(demo_interventions),
            )

            demo_tradeoff = pd.DataFrame(
                [
                    {
                        "Measure": "Net electricity",
                        "Normal IQL": float(
                            demo_normal["net_electricity_consumption"]
                        ),
                        "Guarded IQL": float(
                            demo_guarded["net_electricity_consumption"]
                        ),
                    },
                    {
                        "Measure": "Electricity cost",
                        "Normal IQL": float(
                            demo_normal["electricity_cost"]
                        ),
                        "Guarded IQL": float(
                            demo_guarded["electricity_cost"]
                        ),
                    },
                    {
                        "Measure": "Carbon emissions",
                        "Normal IQL": float(
                            demo_normal["carbon_emission"]
                        ),
                        "Guarded IQL": float(
                            demo_guarded["carbon_emission"]
                        ),
                    },
                    {
                        "Measure": "Peak electricity",
                        "Normal IQL": float(
                            demo_normal["peak_net_electricity"]
                        ),
                        "Guarded IQL": float(
                            demo_guarded["peak_net_electricity"]
                        ),
                    },
                ]
            )

            demo_tradeoff["Change (%)"] = (
                (
                    demo_tradeoff["Guarded IQL"]
                    - demo_tradeoff["Normal IQL"]
                )
                / demo_tradeoff["Normal IQL"]
                * 100.0
            )

            st.dataframe(
                demo_tradeoff.round(3),
                width="stretch",
                hide_index=True,
            )

            st.info(
                f"The supervisory guardrail reduces worst-building overheating "
                f"discomfort by {demo_reduction:.1f}% relative to normal IQL. "
                "This improvement is accompanied by increased resource use, "
                "making the comfort-versus-resource trade-off explicit."
            )

    if not demo_guardrail_ready:
        st.warning("Preserved normal-versus-guarded IQL evidence is unavailable.")

    # -------------------------------------------------------------
    # 5. Engineering decision
    # -------------------------------------------------------------
    st.markdown("### 5. Engineering decision and acceptance")

    st.write(
        "The application does not select a controller from one performance "
        "metric. The Operator Decision Support workflow applies editable "
        "scenario-specific limits to energy, cost, carbon, peak demand, and "
        "worst-building discomfort."
    )

    st.code(
        """Measured controller evidence
        |
        v
Operator-defined engineering limits
        |
        v
PASS / WARN / FAIL / NOT EVALUATED
        |
        v
Conservative overall screening
        |
        v
Controller verification record""",
        language="text",
    )

    st.caption(
        "Any FAIL produces an overall FAIL; otherwise WARN is retained if "
        "present. BasicRBC comfort remains NOT EVALUATED because equivalent "
        "comfort evidence is unavailable."
    )

    # -------------------------------------------------------------
    # 6. Verification evidence
    # -------------------------------------------------------------
    st.markdown("### 6. Verification and reproducibility")

    manifest_controller_count = "Unavailable"
    manifest_controlled_count = "Unavailable"
    manifest_evidence_count = "Unavailable"

    if experiment_manifest is not None:
        manifest_controller_count = str(
            experiment_manifest["controller_artifacts"][
                "available_saved_controller_count"
            ]
        )
        manifest_controlled_count = str(
            experiment_manifest["controller_artifacts"][
                "controlled_evaluation_artifact_count"
            ]
        )
        manifest_evidence_count = str(
            len(experiment_manifest["preserved_evidence"])
        )

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Saved controllers", manifest_controller_count)
    v2.metric("Controlled artifacts", manifest_controlled_count)
    v3.metric("Hashed evidence files", manifest_evidence_count)
    v4.metric("Repository revision", evidence_identity["revision"])

    if (
        guard_events is not None
        and demo_guardrail_ready
        and len(guard_events) == demo_interventions
    ):
        st.success(
            f"Verification check: all {demo_interventions} reported guardrail "
            "interventions have corresponding preserved event records."
        )

    st.write(
        "The experiment manifest preserves SHA-256 identities for controller "
        "artifacts, engineering evidence, and verification tests. The application "
        "also provides downloadable evaluation and verification records."
    )

    st.warning(
        "Validation boundary: this demonstration establishes reproducible "
        "simulation-based engineering evaluation in CityLearn. It does not "
        "establish physical-building safety, certify a controller for deployment, "
        "or replace BMS integration, commissioning, operator override, and field "
        "validation."
    )

    st.success(
        "Demonstrated engineering workflow: logged data → offline controller → "
        "fresh evaluation → multi-KPI assessment → failure detection → supervisory "
        "mitigation → acceptance screening → preserved verification evidence."
    )

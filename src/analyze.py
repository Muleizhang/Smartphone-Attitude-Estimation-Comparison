from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
PROCESSED_DIR = REPORT_DIR / "processed"


GYRO_COLUMNS = {
    "Time (s)": "t",
    "Gyroscope x (rad/s)": "wx",
    "Gyroscope y (rad/s)": "wy",
    "Gyroscope z (rad/s)": "wz",
}

MAG_COLUMNS = {
    "Time (s)": "t",
    "Magnetic field x (µT)": "bx",
    "Magnetic field y (µT)": "by",
    "Magnetic field z (µT)": "bz",
}


def ensure_dirs() -> None:
    for path in (FIG_DIR, TABLE_DIR, PROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_experiment(exp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    gyro = pd.read_csv(DATA_DIR / exp / "Gyroscope.csv").rename(columns=GYRO_COLUMNS)
    mag = pd.read_csv(DATA_DIR / exp / "Magnetometer.csv").rename(columns=MAG_COLUMNS)
    gyro = gyro.loc[:, ["t", "wx", "wy", "wz"]].sort_values("t").reset_index(drop=True)
    mag = mag.loc[:, ["t", "bx", "by", "bz"]].sort_values("t").reset_index(drop=True)
    return gyro, mag


def load_device_model() -> str:
    device = pd.read_csv(DATA_DIR / "exp1" / "meta" / "device.csv")
    model = device.loc[device["property"] == "deviceModel", "value"].iloc[0]
    brand = device.loc[device["property"] == "deviceBrand", "value"].iloc[0]
    return f"{brand} {model}"


def load_time_meta(exp: str) -> dict[str, str]:
    meta = pd.read_csv(DATA_DIR / exp / "meta" / "time.csv")
    return {
        "start": meta.loc[meta["event"] == "START", "system time text"].iloc[0],
        "pause": meta.loc[meta["event"] == "PAUSE", "system time text"].iloc[0],
    }


def sample_rate(df: pd.DataFrame) -> float:
    return (len(df) - 1) / (df["t"].iloc[-1] - df["t"].iloc[0])


def integrate_trapezoid(t: pd.Series, y: pd.Series | np.ndarray) -> np.ndarray:
    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    increments = 0.5 * (yy[1:] + yy[:-1]) * np.diff(tt)
    return np.r_[0.0, np.cumsum(increments)]


def heading_from_magnetometer(mag: pd.DataFrame, unwrap: bool = True) -> np.ndarray:
    # Sign is chosen to match the positive z-axis gyro convention in these data.
    heading = -np.arctan2(mag["by"].to_numpy(), mag["bx"].to_numpy())
    return np.unwrap(heading) if unwrap else heading


def wrap_to_180(degrees: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(degrees) + 180.0) % 360.0 - 180.0


def circular_mean_deg(values_deg: np.ndarray) -> float:
    angles = np.deg2rad(values_deg)
    return math.degrees(math.atan2(np.sin(angles).mean(), np.cos(angles).mean()))


def circular_std_deg(values_deg: np.ndarray) -> float:
    mean = circular_mean_deg(values_deg)
    return float(np.std(wrap_to_180(values_deg - mean), ddof=1))


def window_values(df: pd.DataFrame, column: str, start: float, end: float) -> np.ndarray:
    values = df.loc[(df["t"] >= start) & (df["t"] <= end), column].to_numpy()
    if len(values) == 0:
        raise ValueError(f"empty window for {column}: {start} to {end}")
    return values


def central_window(start: float, end: float, fraction: float = 0.60) -> tuple[float, float]:
    mid = 0.5 * (start + end)
    half_width = 0.5 * (end - start) * fraction
    return mid - half_width, mid + half_width


def detect_stationary_segments(
    gyro: pd.DataFrame,
    threshold_rad_s: float = 0.03,
    smooth_s: float = 0.20,
    min_duration_s: float = 0.50,
    merge_gap_s: float = 0.80,
) -> list[tuple[float, float]]:
    fs = sample_rate(gyro)
    window = max(3, int(round(fs * smooth_s)))
    smoothed = (
        gyro["wz"]
        .abs()
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    t = gyro["t"].to_numpy()
    stationary = smoothed < threshold_rad_s

    raw_segments: list[tuple[float, float]] = []
    start: float | None = None
    for ti, is_stationary in zip(t, stationary):
        if is_stationary and start is None:
            start = float(ti)
        elif not is_stationary and start is not None:
            end = float(ti)
            if end - start >= min_duration_s:
                raw_segments.append((start, end))
            start = None
    if start is not None and float(t[-1]) - start >= min_duration_s:
        raw_segments.append((start, float(t[-1])))

    merged: list[tuple[float, float]] = []
    for start, end in raw_segments:
        if merged and start - merged[-1][1] <= merge_gap_s:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def latex_num(value: float, digits: int = 2) -> str:
    if math.isnan(value):
        return "--"
    return f"{value:.{digits}f}"


def write_latex_table(path: Path, tabular: str, caption: str, label: str) -> None:
    path.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                tabular,
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_metadata_table(summary: pd.DataFrame, device_model: str) -> None:
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            f"{row.exp} & {row.duration_s:.2f} & {row.gyro_samples} & "
            f"{row.gyro_fs_hz:.1f} & {row.mag_samples} & {row.mag_fs_hz:.1f} \\\\"
        )
    tabular = "\n".join(
        [
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"实验 & 时长/s & 陀螺仪样本 & 陀螺仪/Hz & 磁力计样本 & 磁力计/Hz \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
            rf"{{\vspace{{0.4em}}\small 采集设备：{device_model}。\par}}",
        ]
    )
    write_latex_table(TABLE_DIR / "metadata.tex", tabular, "数据集基本信息", "tab:metadata")


def build_exp1_table(metrics: dict[str, float]) -> None:
    rows = [
        ("陀螺仪 $\\omega_z$ 零偏", metrics["gyro_bias_rad_s"], "rad/s", 6),
        ("陀螺仪等效漂移率", metrics["gyro_drift_deg_min"], r"$^\circ$/min", 3),
        ("未校正积分终值", metrics["gyro_raw_final_deg"], r"$^\circ$", 3),
        ("零偏校正后积分终值", metrics["gyro_corrected_final_deg"], r"$^\circ$", 3),
        ("磁力计航向标准差", metrics["mag_heading_std_deg"], r"$^\circ$", 3),
        ("磁场模长均值", metrics["mag_norm_mean_uT"], r"$\mu$T", 2),
        ("磁场模长标准差", metrics["mag_norm_std_uT"], r"$\mu$T", 2),
    ]
    body = [
        f"{name} & {value:.{digits}f} & {unit} \\\\" for name, value, unit, digits in rows
    ]
    tabular = "\n".join(
        [
            r"\begin{tabular}{lrl}",
            r"\toprule",
            r"指标 & 数值 & 单位 \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_latex_table(TABLE_DIR / "exp1_static.tex", tabular, "实验 1：静态漂移统计", "tab:exp1")


def build_exp2_table(results: pd.DataFrame) -> None:
    rows = []
    for row in results.itertuples(index=False):
        rows.append(
            f"{row.expected_deg:.0f} & "
            f"{row.gyro_deg:.2f} & {row.gyro_error_deg:+.2f} & {row.gyro_std_deg:.2f} & "
            f"{row.mag_deg:.2f} & {row.mag_error_deg:+.2f} & {row.mag_std_deg:.2f} \\\\"
        )
    tabular = "\n".join(
        [
            r"\begin{tabular}{rrrrrrr}",
            r"\toprule",
            r"参考角/$^\circ$ & 陀螺仪/$^\circ$ & 误差/$^\circ$ & 稳定段$\sigma$ & 磁力计/$^\circ$ & 误差/$^\circ$ & 稳定段$\sigma$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_latex_table(TABLE_DIR / "exp2_angles.tex", tabular, "实验 2：定角度旋转结果", "tab:exp2")


def build_exp3_table(metrics: dict[str, float]) -> None:
    rows = [
        ("陀螺仪积分总角度", metrics["gyro_net_deg"], r"$^\circ$", 2),
        ("最近整圈数", metrics["gyro_nearest_turns"], "圈", 0),
        ("陀螺仪回零残余", metrics["gyro_residual_deg"], r"$^\circ$", 2),
        ("磁力计起终航向差", metrics["mag_residual_deg"], r"$^\circ$", 2),
        ("磁场模长均值", metrics["mag_norm_mean_uT"], r"$\mu$T", 2),
        ("磁场模长最大值", metrics["mag_norm_max_uT"], r"$\mu$T", 1),
        ("强干扰样本占比", metrics["mag_norm_over_100_ratio"], r"\%", 2),
    ]
    body = [
        f"{name} & {value:.{digits}f} & {unit} \\\\" for name, value, unit, digits in rows
    ]
    tabular = "\n".join(
        [
            r"\begin{tabular}{lrl}",
            r"\toprule",
            r"指标 & 数值 & 单位 \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_latex_table(TABLE_DIR / "exp3_return.tex", tabular, "实验 3：回归原点结果", "tab:exp3")


def plot_exp1(gyro: pd.DataFrame, mag: pd.DataFrame, bias: float) -> None:
    gyro_raw = np.rad2deg(integrate_trapezoid(gyro["t"], gyro["wz"]))
    gyro_corrected = np.rad2deg(integrate_trapezoid(gyro["t"], gyro["wz"] - bias))
    mag_heading = np.rad2deg(heading_from_magnetometer(mag))
    mag_relative = mag_heading - np.mean(mag_heading)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)
    axes[0].plot(gyro["t"], gyro_raw, label="gyro raw integral", lw=1.5)
    axes[0].plot(gyro["t"], gyro_corrected, label="gyro bias corrected", lw=1.2)
    axes[0].set_ylabel("Yaw / deg")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(mag["t"], mag_relative, color="#a23b72", lw=1.0)
    axes[1].axhline(0.0, color="black", lw=0.8, alpha=0.6)
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("Mag. heading residual / deg")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp1_static.pdf")
    plt.close(fig)


def plot_exp2(
    gyro: pd.DataFrame,
    mag: pd.DataFrame,
    bias: float,
    segments: list[tuple[float, float]],
    results: pd.DataFrame,
) -> None:
    gyro_yaw = np.rad2deg(integrate_trapezoid(gyro["t"], gyro["wz"] - bias))
    mag_heading = np.rad2deg(heading_from_magnetometer(mag))

    first_start, first_end = central_window(*segments[0])
    gyro_zero = window_values(pd.DataFrame({"t": gyro["t"], "yaw": gyro_yaw}), "yaw", first_start, first_end).mean()
    mag_zero = window_values(pd.DataFrame({"t": mag["t"], "yaw": mag_heading}), "yaw", first_start, first_end).mean()

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(gyro["t"], gyro_yaw - gyro_zero, label="gyro integral", lw=1.4)
    ax.plot(mag["t"], mag_heading - mag_zero, label="magnetometer", lw=1.0, alpha=0.85)
    for start, end in segments:
        c0, c1 = central_window(start, end)
        ax.axvspan(c0, c1, color="0.85", alpha=0.35, lw=0)
    ax.scatter(results["segment_center_s"], results["gyro_deg"], color="#1f77b4", s=28, zorder=3)
    ax.scatter(results["segment_center_s"], results["mag_deg"], color="#a23b72", s=28, zorder=3)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Relative yaw / deg")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp2_step_response.pdf")
    plt.close(fig)

    x = np.arange(len(results))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(x - width / 2, results["gyro_error_deg"], width, label="gyro", color="#1f77b4")
    ax.bar(x + width / 2, results["mag_error_deg"], width, label="magnetometer", color="#a23b72")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x, [f"{v:.0f}" for v in results["expected_deg"]])
    ax.set_xlabel("Reference angle / deg")
    ax.set_ylabel("Error / deg")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp2_errors.pdf")
    plt.close(fig)


def plot_exp3(gyro: pd.DataFrame, mag: pd.DataFrame, bias: float) -> None:
    gyro_yaw = np.rad2deg(integrate_trapezoid(gyro["t"], gyro["wz"] - bias))
    gyro_wrapped = wrap_to_180(gyro_yaw - gyro_yaw[0])
    mag_heading = np.rad2deg(heading_from_magnetometer(mag))
    mag_wrapped = wrap_to_180(mag_heading - mag_heading[0])
    mag_norm = np.sqrt(mag["bx"] ** 2 + mag["by"] ** 2 + mag["bz"] ** 2)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, height_ratios=[2.0, 1.0])
    axes[0].plot(gyro["t"], gyro_wrapped, label="gyro integral (wrapped)", lw=1.1)
    axes[0].plot(mag["t"], mag_wrapped, label="magnetometer (wrapped)", lw=1.0, alpha=0.85)
    axes[0].set_ylabel("Relative yaw / deg")
    axes[0].set_ylim(-190, 190)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(mag["t"], mag_norm, color="#c77d00", lw=1.0)
    axes[1].axhline(100.0, color="black", ls="--", lw=0.8, alpha=0.7)
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("|B| / microtesla")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp3_return.pdf")
    plt.close(fig)


def analyze() -> dict[str, object]:
    ensure_dirs()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    data = {exp: load_experiment(exp) for exp in ("exp1", "exp2", "exp3")}
    device_model = load_device_model()

    summary_rows = []
    for exp, (gyro, mag) in data.items():
        time_meta = load_time_meta(exp)
        summary_rows.append(
            {
                "exp": exp,
                "start_time": time_meta["start"],
                "pause_time": time_meta["pause"],
                "duration_s": max(gyro["t"].iloc[-1], mag["t"].iloc[-1]),
                "gyro_samples": len(gyro),
                "gyro_fs_hz": sample_rate(gyro),
                "mag_samples": len(mag),
                "mag_fs_hz": sample_rate(mag),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(PROCESSED_DIR / "sample_summary.csv", index=False)
    build_metadata_table(summary, device_model)

    exp1_gyro, exp1_mag = data["exp1"]
    gyro_bias = float(exp1_gyro["wz"].mean())
    gyro_raw = np.rad2deg(integrate_trapezoid(exp1_gyro["t"], exp1_gyro["wz"]))
    gyro_corrected = np.rad2deg(integrate_trapezoid(exp1_gyro["t"], exp1_gyro["wz"] - gyro_bias))
    exp1_mag_heading = np.rad2deg(heading_from_magnetometer(exp1_mag))
    exp1_mag_norm = np.sqrt(exp1_mag["bx"] ** 2 + exp1_mag["by"] ** 2 + exp1_mag["bz"] ** 2)
    exp1_metrics = {
        "gyro_bias_rad_s": gyro_bias,
        "gyro_drift_deg_min": math.degrees(gyro_bias) * 60.0,
        "gyro_raw_final_deg": float(gyro_raw[-1]),
        "gyro_corrected_final_deg": float(gyro_corrected[-1]),
        "mag_heading_std_deg": float(np.std(exp1_mag_heading, ddof=1)),
        "mag_norm_mean_uT": float(exp1_mag_norm.mean()),
        "mag_norm_std_uT": float(exp1_mag_norm.std(ddof=1)),
    }
    pd.DataFrame([exp1_metrics]).to_csv(PROCESSED_DIR / "exp1_static_metrics.csv", index=False)
    build_exp1_table(exp1_metrics)
    plot_exp1(exp1_gyro, exp1_mag, gyro_bias)

    exp2_gyro, exp2_mag = data["exp2"]
    exp2_segments = detect_stationary_segments(exp2_gyro)[:5]
    if len(exp2_segments) != 5:
        raise RuntimeError(f"expected 5 stationary angle segments in exp2, got {len(exp2_segments)}")
    exp2_gyro_yaw = np.rad2deg(integrate_trapezoid(exp2_gyro["t"], exp2_gyro["wz"] - gyro_bias))
    exp2_mag_heading = np.rad2deg(heading_from_magnetometer(exp2_mag))
    exp2_gyro_tmp = pd.DataFrame({"t": exp2_gyro["t"], "yaw": exp2_gyro_yaw})
    exp2_mag_tmp = pd.DataFrame({"t": exp2_mag["t"], "yaw": exp2_mag_heading})
    zero_start, zero_end = central_window(*exp2_segments[0])
    gyro_zero = window_values(exp2_gyro_tmp, "yaw", zero_start, zero_end).mean()
    mag_zero = window_values(exp2_mag_tmp, "yaw", zero_start, zero_end).mean()

    exp2_rows = []
    for expected, (start, end) in zip([0, 90, 180, 270, 360], exp2_segments):
        win_start, win_end = central_window(start, end)
        gyro_vals = window_values(exp2_gyro_tmp, "yaw", win_start, win_end) - gyro_zero
        mag_vals = window_values(exp2_mag_tmp, "yaw", win_start, win_end) - mag_zero
        gyro_mean = float(gyro_vals.mean())
        mag_mean = float(mag_vals.mean())
        exp2_rows.append(
            {
                "expected_deg": float(expected),
                "segment_start_s": start,
                "segment_end_s": end,
                "segment_center_s": 0.5 * (win_start + win_end),
                "gyro_deg": gyro_mean,
                "gyro_error_deg": gyro_mean - expected,
                "gyro_std_deg": float(gyro_vals.std(ddof=1)),
                "mag_deg": mag_mean,
                "mag_error_deg": mag_mean - expected,
                "mag_std_deg": float(mag_vals.std(ddof=1)),
            }
        )
    exp2_results = pd.DataFrame(exp2_rows)
    exp2_results.to_csv(PROCESSED_DIR / "exp2_segment_angles.csv", index=False)
    build_exp2_table(exp2_results)
    plot_exp2(exp2_gyro, exp2_mag, gyro_bias, exp2_segments, exp2_results)

    exp3_gyro, exp3_mag = data["exp3"]
    exp3_gyro_yaw = np.rad2deg(integrate_trapezoid(exp3_gyro["t"], exp3_gyro["wz"] - gyro_bias))
    exp3_mag_heading_wrapped = np.rad2deg(heading_from_magnetometer(exp3_mag, unwrap=False))
    exp3_mag_norm = np.sqrt(exp3_mag["bx"] ** 2 + exp3_mag["by"] ** 2 + exp3_mag["bz"] ** 2)

    start_window_s = 2.0
    end_window_s = 2.0
    gyro_start_vals = exp3_gyro_yaw[exp3_gyro["t"] <= exp3_gyro["t"].iloc[0] + start_window_s]
    gyro_end_vals = exp3_gyro_yaw[exp3_gyro["t"] >= exp3_gyro["t"].iloc[-1] - end_window_s]
    gyro_net = float(gyro_end_vals.mean() - gyro_start_vals.mean())
    nearest_turns = int(round(gyro_net / 360.0))
    gyro_residual = gyro_net - 360.0 * nearest_turns

    mag_start_vals = exp3_mag_heading_wrapped[
        exp3_mag["t"] <= exp3_mag["t"].iloc[0] + start_window_s
    ]
    mag_end_vals = exp3_mag_heading_wrapped[
        exp3_mag["t"] >= exp3_mag["t"].iloc[-1] - end_window_s
    ]
    mag_start_mean = circular_mean_deg(mag_start_vals)
    mag_end_mean = circular_mean_deg(mag_end_vals)
    mag_residual = float(wrap_to_180(mag_end_mean - mag_start_mean))
    exp3_metrics = {
        "gyro_net_deg": gyro_net,
        "gyro_nearest_turns": float(nearest_turns),
        "gyro_residual_deg": gyro_residual,
        "gyro_start_std_deg": float(np.std(gyro_start_vals, ddof=1)),
        "gyro_end_std_deg": float(np.std(gyro_end_vals, ddof=1)),
        "mag_residual_deg": mag_residual,
        "mag_start_std_deg": circular_std_deg(mag_start_vals),
        "mag_end_std_deg": circular_std_deg(mag_end_vals),
        "mag_norm_mean_uT": float(exp3_mag_norm.mean()),
        "mag_norm_std_uT": float(exp3_mag_norm.std(ddof=1)),
        "mag_norm_max_uT": float(exp3_mag_norm.max()),
        "mag_norm_max_time_s": float(exp3_mag.loc[exp3_mag_norm.idxmax(), "t"]),
        "mag_norm_over_100_ratio": float((exp3_mag_norm > 100.0).mean() * 100.0),
    }
    pd.DataFrame([exp3_metrics]).to_csv(PROCESSED_DIR / "exp3_return_metrics.csv", index=False)
    build_exp3_table(exp3_metrics)
    plot_exp3(exp3_gyro, exp3_mag, gyro_bias)

    results = {
        "device_model": device_model,
        "summary": summary.to_dict(orient="records"),
        "exp1": exp1_metrics,
        "exp2": exp2_results.to_dict(orient="records"),
        "exp3": exp3_metrics,
    }
    (PROCESSED_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


if __name__ == "__main__":
    analyze()
    print(f"analysis outputs written to {REPORT_DIR}")

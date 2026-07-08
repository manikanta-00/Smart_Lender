"""
eda.py
------
"3. Data Visualization & Analysis" from the project brief, implemented with
hand-rolled SVG generation so we don't need matplotlib / seaborn.

Produces:
  - Count plots   (Gender, Married, Education, Self_Employed, Property_Area, Loan_Status)
  - Distribution plot (ApplicantIncome histogram)
  - Bar chart      (Approval rate by Property_Area)

All charts are written as inline SVG into static/eda/eda_report.html.
"""

import os
from collections import Counter

PALETTE = ["#4f8cff", "#22c58b", "#ff8a5c", "#a06bff", "#ffca3a", "#ff5c8a"]


def _svg_bar_chart(title, labels, values, width=520, height=320, color_list=None):
    color_list = color_list or PALETTE
    max_val = max(values) if values else 1
    max_val = max_val if max_val > 0 else 1
    padding_left = 60
    padding_bottom = 70
    padding_top = 40
    chart_w = width - padding_left - 30
    chart_h = height - padding_top - padding_bottom
    n = len(labels)
    bar_width = chart_w / n * 0.6
    gap = chart_w / n

    bars = []
    for i, (label, value) in enumerate(zip(labels, values)):
        bar_h = (value / max_val) * chart_h
        x = padding_left + i * gap + (gap - bar_width) / 2
        y = padding_top + (chart_h - bar_h)
        color = color_list[i % len(color_list)]
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" '
            f'rx="4" fill="{color}"><title>{label}: {value}</title></rect>'
        )
        bars.append(
            f'<text x="{x + bar_width/2:.1f}" y="{y - 6:.1f}" font-size="12" '
            f'text-anchor="middle" fill="#333">{value}</text>'
        )
        bars.append(
            f'<text x="{x + bar_width/2:.1f}" y="{height - padding_bottom + 20:.1f}" '
            f'font-size="12" text-anchor="middle" fill="#333" transform="rotate(0 '
            f'{x + bar_width/2:.1f} {height - padding_bottom + 20:.1f})">{label}</text>'
        )

    axis = (
        f'<line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" '
        f'y2="{padding_top + chart_h}" stroke="#999" stroke-width="1"/>'
        f'<line x1="{padding_left}" y1="{padding_top + chart_h}" x2="{width - 20}" '
        f'y2="{padding_top + chart_h}" stroke="#999" stroke-width="1"/>'
    )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#fff;border-radius:10px">'
        f'<text x="{width/2}" y="24" font-size="16" font-weight="600" '
        f'text-anchor="middle" fill="#1a1a1a">{title}</text>'
        f'{axis}{"".join(bars)}</svg>'
    )
    return svg


def _count_plot(rows, column, title=None):
    counts = Counter(r.get(column, "Unknown") or "Unknown" for r in rows)
    labels = list(counts.keys())
    values = [counts[l] for l in labels]
    return _svg_bar_chart(title or f"Count Plot: {column}", labels, values)


def _distribution_plot(rows, column, bins=10, title=None):
    values = []
    for r in rows:
        try:
            values.append(float(r.get(column)))
        except (TypeError, ValueError):
            continue
    if not values:
        return "<p>No data</p>"
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    bin_width = span / bins
    bucket_counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / bin_width))
        bucket_counts[idx] += 1
    labels = [f"{int(lo + i*bin_width)}-{int(lo + (i+1)*bin_width)}" for i in range(bins)]
    return _svg_bar_chart(title or f"Distribution: {column}", labels, bucket_counts,
                           width=680, color_list=["#4f8cff"])


def _approval_rate_by_group(rows, column, title=None):
    groups = {}
    for r in rows:
        key = r.get(column, "Unknown") or "Unknown"
        groups.setdefault(key, [0, 0])  # [approved, total]
        groups[key][1] += 1
        if r.get("Loan_Status") == "Y":
            groups[key][0] += 1
    labels = list(groups.keys())
    rates = [round(100 * groups[l][0] / groups[l][1], 1) if groups[l][1] else 0 for l in labels]
    return _svg_bar_chart(title or f"Approval Rate % by {column}", labels, rates,
                           color_list=["#22c58b"])


def generate_eda_report(rows, output_path):
    charts = []
    charts.append(("Gender Distribution", _count_plot(rows, "Gender")))
    charts.append(("Education Distribution", _count_plot(rows, "Education")))
    charts.append(("Property Area Distribution", _count_plot(rows, "Property_Area")))
    charts.append(("Loan Status (Target) Distribution", _count_plot(rows, "Loan_Status")))
    charts.append(("Applicant Income Distribution", _distribution_plot(rows, "ApplicantIncome")))
    charts.append(("Loan Amount Distribution", _distribution_plot(rows, "LoanAmount")))
    charts.append(("Approval Rate by Property Area", _approval_rate_by_group(rows, "Property_Area")))
    charts.append(("Approval Rate by Education", _approval_rate_by_group(rows, "Education")))

    cards = "".join(
        f'<div class="chart-card"><h3>{name}</h3>{svg}</div>' for name, svg in charts
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Smart Lender - EDA Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6fb; margin:0; padding:30px; }}
  h1 {{ color:#1a1a1a; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(480px,1fr)); gap:24px; }}
  .chart-card {{ background:#fff; border-radius:12px; padding:16px; box-shadow:0 2px 10px rgba(0,0,0,0.06); }}
  .chart-card h3 {{ margin:4px 0 12px 4px; color:#333; font-size:15px; }}
  a.back {{ display:inline-block; margin-bottom:20px; color:#4f8cff; text-decoration:none; font-weight:600; }}
</style>
</head>
<body>
<a class="back" href="/">&larr; Back to Home</a>
<h1>Exploratory Data Analysis</h1>
<p>Auto-generated count plots, distribution plots, and bar charts from the loan dataset (pure-SVG, no matplotlib/seaborn required).</p>
<div class="grid">
{cards}
</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[eda] Report written to {output_path}")

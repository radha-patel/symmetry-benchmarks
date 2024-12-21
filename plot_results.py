import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from collections import defaultdict
import re
import os

CHARTS_DIRECTORY = "charts/"

def use_suitesparse_name(result):
    return result["matrix"]

def use_sparsity(result):
    s = result["sparsity"]
    return f"{s}"

def use_sparsity_rank(result):
    s = result["sparsity"]
    r = result["rank"]
    return f"{s, r}"


def all_formats_chart(results_filepath, get_matrix_name, reference, optimized, methods, colors, title, y_label, x_label, legend_labels, ordered = True, width=12, expected_speedup=0):
    results = json.load(open(results_filepath, 'r'))
    data = defaultdict(lambda: defaultdict(int))

    for result in results:
        mtx = get_matrix_name(result)
        method = result["method"]
        time = result["time"]
        data[mtx][method] = time

    for mtx, times in data.items():
        ref_time = times[reference]
        for method, time in times.items():
            times[method] = ref_time / time

    if ordered:
        ordered_data = sorted(data.items(), key = lambda mtx_results: mtx_results[1][optimized], reverse=True)
    else:
        ordered_data = data.items()

    all_data = defaultdict(list)
    for i, (mtx, times) in enumerate(ordered_data):
        for method, time in times.items():
            all_data[method].append(time)

    ordered_mtxs = [mtx for mtx, _ in ordered_data]
    short_mtxs = [mtx.rsplit('/',1)[-1] for mtx in ordered_mtxs]
    make_grouped_bar_chart(methods, short_mtxs, all_data, colors=colors, title=title, y_label=y_label, x_label=x_label, legend_labels=legend_labels, plot_width=width, expected_speedup=expected_speedup)


def make_grouped_bar_chart(labels, x_axis, data, colors = None, labeled_groups = [], title = "", y_label = "", x_label = "", bar_labels_dict={}, legend_labels=None, reference_label = "", ref_line = True, plot_width = 12, expected_speedup=0):
    horizontal_scale = 1

    x = np.arange(0, len(data[labels[0]]) * horizontal_scale, horizontal_scale)
    width = 0.3
    max_height = 0
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(plot_width, 3.5))
    for label in labels:
        label_data = data[label]
        max_height = max(max_height, max(label_data))
        offset = width * multiplier
        rects = ax.bar(x + offset, label_data, width, label=label, color=colors[label])
        bar_labels = bar_labels_dict[label] if (label in bar_labels_dict) else [round(float(val), 2) if label in labeled_groups else "" for val in label_data]
        ax.bar_label(rects, padding=0, labels=bar_labels, fontsize=4, rotation=90)
        multiplier += 1

    if y_label:
        ax.set_ylabel(y_label, fontsize=14)
    # ax.set_title(title, fontsize=12, fontfamily=fontfamily)
    if x_label:
        ax.set_xlabel(x_label, fontsize=14)
    ax.set_xticks(x + width * (len(labels) - 1)/2, x_axis)
    ax.tick_params(axis='x', which='major', labelsize=11, labelrotation=60)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha="right")
    ax.tick_params(axis='y', labelsize=10)
    if legend_labels:
        ax.legend(legend_labels, loc='upper right', ncols=1, fontsize=12)
    ax.set_ylim(0, max_height + 0.05 * max_height)

        # Adjusting x-axis limits to make bars go to the edges
    ax.set_xlim(-0.5, (len(x_axis) - 0.5 + width * len(labels)) * horizontal_scale)

    if ref_line:
        plt.plot([-1, len(x_axis)], [1, 1], linestyle='--', color="tab:red", linewidth=2, label=reference_label)
    if expected_speedup:
        plt.plot([-1, len(x_axis)], [expected_speedup, expected_speedup], linestyle='--', color="purple", linewidth=1.25, label=reference_label)

    fig_file = title.lower().replace(" ", "_") + ".png"
    plt.savefig(CHARTS_DIRECTORY + fig_file, dpi=200, bbox_inches="tight")
    plt.close()


if os.path.exists("ssymv_results.json"):
    all_formats_chart("ssymv_results.json", use_suitesparse_name, "ssymv_finch_ref", "ssymv_finch_opt",
                    ["ssymv_finch_opt", "ssymv_taco"],#, "ssymv_mkl"],  
                    {"ssymv_finch_opt": "tab:blue", "ssymv_taco": "tab:orange"}, #, "ssymv_mkl": "tab:green"},
                    "SSYMV Performance",
                    "Speedup",
                    "Matrix Name",
                    ["SySTeC", "TACO"],#, "MKL"],
                    expected_speedup = 1, width=16)

if os.path.exists("bellman_results.json"):
    all_formats_chart("bellman_results.json", use_suitesparse_name, "bellman_finch_ref", "bellman_finch_opt",
                    ["bellman_finch_opt", "bellman_taco"],#, "bellman_mkl"],  
                    {"bellman_finch_opt": "tab:blue", "bellman_taco": "tab:orange"}, #, "bellman_mkl": "tab:green"},
                    "Bellman-Ford Step Performance",
                    "Speedup",
                    "Matrix Name",
                    ["SySTeC", "TACO"],#, "MKL"],
                    expected_speedup = 1, width=16)

if os.path.exists("kernels/ssyrk/ssyrk_results.json"):
    all_formats_chart("kernels/ssyrk/ssyrk_results.json", use_suitesparse_name, "ssyrk_ref", "ssyrk_opt",
                    ["ssyrk_opt"],  
                    {"ssyrk_opt": "tab:blue"},
                    "SSYRK Performance",
                    "Speedup",
                    "Matrix Name",
                    ["SySTeC"],
                    expected_speedup = 2, width=16)

if os.path.exists("syprd_results.json"):
    all_formats_chart("syprd_results.json", use_suitesparse_name, "syprd_ref", "syprd_opt",
                    ["syprd_opt", "syprd_taco"],  
                    {"syprd_opt": "tab:blue", "syprd_taco": "tab:orange"},
                    "SYPRD Performance",
                    "Speedup",
                    "Matrix Name",
                    ["SySTeC", "TACO"],
                    expected_speedup = 2, width=16)

if os.path.exists("ttm_results.json"):
    all_formats_chart("ttm_results.json", use_sparsity_rank, "ttm_finch_ref", "ttm_finch_opt",
                    ["ttm_finch_opt", "ttm_taco"],  
                    {"ttm_finch_opt": "tab:blue", "ttm_taco": "tab:orange"},
                    "TTM Performance",
                    "Speedup",
                    "Sparsity, Numerical Rank",
                    ["SySTeC", "TACO"],
                    ordered = False,
                    width = 8,
                    expected_speedup = 2)

if os.path.exists("mttkrp_dim3_results.json"):
    all_formats_chart("mttkrp_dim3_results.json", use_sparsity_rank, "mttkrp_finch_ref", "mttkrp_finch_opt",
                    ["mttkrp_finch_opt", "mttkrp_taco", "mttkrp_splatt"],  
                    {"mttkrp_finch_opt": "tab:blue", "mttkrp_taco": "tab:orange", "mttkrp_splatt": "tab:green"},
                    "3D MTTKRP Performance",
                    "Speedup",
                    "", #"Sparsity, Numerical Rank",
                    ["SySTeC", "TACO", "SPLATT"], 
                    ordered = False,
                    width = 5,
                    expected_speedup = 2)

if os.path.exists("mttkrp_dim4_results.json"):
    all_formats_chart("mttkrp_dim4_results.json", use_sparsity_rank, "mttkrp_finch_ref", "mttkrp_finch_opt",
                    ["mttkrp_finch_opt", "mttkrp_taco", "mttkrp_splatt"],  
                    {"mttkrp_finch_opt": "tab:blue", "mttkrp_taco": "tab:orange", "mttkrp_splatt": "tab:green"},
                    "4D MTTKRP Performance",
                    "Speedup",
                    "", #"Sparsity, Numerical Rank",
                    ["SySTeC", "TACO", "SPLATT"],
                    ordered = False,
                    width = 5,
                    expected_speedup = 6)

if os.path.exists("mttkrp_dim5_results.json"):
    all_formats_chart("mttkrp_dim5_results.json", use_sparsity_rank, "mttkrp_finch_ref", "mttkrp_finch_opt",
                    ["mttkrp_finch_opt", "mttkrp_taco", "mttkrp_splatt"],  
                    {"mttkrp_finch_opt": "tab:blue", "mttkrp_taco": "tab:orange", "mttkrp_splatt": "tab:green"},
                    "5D MTTKRP Performance",
                    "Speedup",
                    "", #"Sparsity, Numerical Rank",
                    ["SySTeC", "TACO", "SPLATT"],
                    ordered = False,
                    width = 5,
                    expected_speedup = 24)

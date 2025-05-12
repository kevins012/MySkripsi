# ✅ FINAL PROGRAM: GUI Time Series Visualizer with NaN Handling, Transform, Highlight, and Trim Timesteps

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import ttk

# Load data
try:
    raw_df = pd.read_csv('csv/dinamic/filter2.csv')
except FileNotFoundError:
    print("File not found: csv/dinamic/raw.csv")
    exit()

feature_cols = [col for col in raw_df.columns if col.startswith('X') or col.startswith('Y')]
unique_labels = sorted(raw_df['Label'].dropna().unique())
unique_sequences = sorted(raw_df['sequence'].dropna().astype(int).unique())

# GUI setup
root = tk.Tk()
root.title("Time Series Visualizer")
root.geometry("1700x950")
root.configure(bg="#f4f4f4")

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", font=("Segoe UI", 11), background="#f4f4f4")
style.configure("TButton", font=("Segoe UI", 10, "bold"))
style.configure("TCombobox", font=("Segoe UI", 10))

main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill='both', expand=True)

frame_left = ttk.LabelFrame(main_frame, text="Select Data", padding=10)
frame_left.grid(row=0, column=0, sticky="n")

frame_mid = ttk.LabelFrame(main_frame, text="Actions", padding=10)
frame_mid.grid(row=0, column=1, padx=20, sticky="n")

frame_right = ttk.LabelFrame(main_frame, text="NaN Handling", padding=10)
frame_right.grid(row=0, column=2, sticky="n")

frame_extra = ttk.LabelFrame(main_frame, text="Transform", padding=10)
frame_extra.grid(row=0, column=3, padx=20, sticky="n")

frame_info = ttk.LabelFrame(main_frame, text="Selected Info (Click to Filter)", padding=10)
frame_info.grid(row=0, column=4, sticky="n")

frame_trim = ttk.LabelFrame(main_frame, text="Trim Timesteps", padding=10)
frame_trim.grid(row=1, column=0, columnspan=2, pady=10)

trim_label = ttk.Label(frame_trim, text="Max timesteps:")
trim_label.pack(side='left', padx=(5, 2))
trim_var = tk.StringVar()
trim_entry = ttk.Entry(frame_trim, width=6, textvariable=trim_var)
trim_entry.pack(side='left', padx=(0, 5))

# Listbox helper
def make_listbox(parent, items, label, height=10):
    ttk.Label(parent, text=label).pack(anchor="w", pady=(5, 2))
    box = tk.Listbox(parent, selectmode='multiple', height=height, exportselection=False)
    for item in items:
        box.insert(tk.END, item)
    box.pack(fill='x')
    return box

feature_listbox = make_listbox(frame_left, feature_cols, "Features (X/Y):", 5)
sequence_listbox = make_listbox(frame_left, unique_sequences, "Sequences:", 5)
label_listbox = make_listbox(frame_left, unique_labels, "Labels:", 5)

nan_methods = {
    "Drop NaN": "drop",
    "Fill 0": "fill_zero",
    "Row Mean": "fill_mean_row",
    "Ffill": "ffill",
    "Bfill": "bfill",
    "Interpolate": "interpolate",
    "Moving Avg": "moving_avg"
}
nan_listbox = make_listbox(frame_right, list(nan_methods.keys()), "NaN Methods:", 8)

reset_timestep = tk.BooleanVar(value=False)
tk.Checkbutton(frame_right, text="Reset timestep", variable=reset_timestep, bg="#f4f4f4").pack(anchor="w")

transform_methods = {
    "Diff1": "diff1",
    
    "Diff2": "diff2",
    "Min-Max": "minmax",
    "Cumsum": "cumsum",
    "Log": "log",
    "EMA": "ema",
    "RollingMed": "rolling_median"
}
transform_listbox = make_listbox(frame_extra, list(transform_methods.keys()), "Transform Methods:", 8)

frame_info_feature = ttk.LabelFrame(frame_info, text="Feature")
frame_info_feature.pack(fill='both')
frame_info_seq = ttk.LabelFrame(frame_info, text="Sequence")
frame_info_seq.pack(fill='both')
frame_info_label = ttk.LabelFrame(frame_info, text="Label")
frame_info_label.pack(fill='both')
frame_info_nan = ttk.LabelFrame(frame_info, text="NaN")
frame_info_nan.pack(fill='both')
frame_info_tx = ttk.LabelFrame(frame_info, text="Transform")
frame_info_tx.pack(fill='both')

# State and style
fig, ax = None, None
line_refs = []
colors = plt.cm.tab10.colors
markers = ['o', '^', 's', 'D', '*', 'x', 'P', 'v']

info_filters = {"Feature": set(), "Sequence": set(), "Label": set(), "NaN": set(), "Transform": set()}

# Helper: evenly trim
# Ensures first and last points are included and evenly distributes middle points
def trim_timestep(x, y, max_len):
    n = len(x)
    if n <= max_len:
        return x, y
    step = (n - 1) / (max_len - 1)
    idx = [round(i * step) for i in range(max_len)]
    return np.arange(1, max_len + 1), y[idx]

# Nan handling and transform
def apply_nan(df, method):
    df = df.copy()
    print(df[:3])
    if method == "drop": df[feature_cols] = df[feature_cols].where(~df[feature_cols].isna(), np.nan)
    elif method == "fill_zero": df[feature_cols] = df[feature_cols].fillna(0)
    elif method == "fill_mean_row": df[feature_cols] = df.groupby(['sequence', 'Label'])[feature_cols].apply(lambda row: row.fillna(row.mean()), axis=1)
    elif method == "ffill": df[feature_cols] = df[feature_cols].fillna(method='ffill')
    elif method == "bfill": df[feature_cols] = df[feature_cols].fillna(method='bfill')
    elif method == "interpolate": df[feature_cols] = df[feature_cols].interpolate(method='linear', limit_direction='both')
    elif method == "moving_avg":df[feature_cols] = (
        df[feature_cols]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=[0,1], drop=True)
    )
    print(df[:3])


    return df

def apply_transform(df, method):
    df = df.copy()

    if method == "diff1":
        df[feature_cols] = df.groupby(['sequence', 'Label'])[feature_cols].diff()

    elif method == "diff2":
        df[feature_cols] = df.groupby(['sequence', 'Label'])[feature_cols].diff().diff()
    elif method == "zscore":
        df[feature_cols] = df.groupby(['sequence', 'Label'])[feature_cols].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
    elif method == "minmax":
        df[feature_cols] = df.groupby(['sequence', 'Label'])[feature_cols].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
    elif method == "cumsum":
        df[feature_cols] = df.groupby(['sequence', 'Label'])[feature_cols].cumsum()
    elif method == "log":
        df[feature_cols] = df[feature_cols].apply(lambda x: np.log1p(x))
    elif method == "ema":
        df[feature_cols] = df.groupby(['sequence', 'Label'])[feature_cols].transform(
            lambda x: x.ewm(span=3, adjust=False).mean()
        )
    elif method == "rolling_median":
        df[feature_cols] = df.groupby(['sequence', 'Label'])[feature_cols].transform(
            lambda x: x.rolling(window=3, min_periods=1).median()
        )

    return df

# Interaktif filter

def update_line_opacity():
    for line, info in line_refs:
        f, l, s, n, t = info
        if (f in info_filters['Feature'] and l in info_filters['Label'] and str(s) in info_filters['Sequence']
                and n in info_filters['NaN'] and t in info_filters['Transform']):
            line.set_alpha(1.0)
            line.set_linewidth(2.0)
        else:
            line.set_alpha(0.15)
            line.set_linewidth(1.0)
    if fig:
        fig.canvas.draw()

def add_toggle_button(frame, group, value):
    if value not in info_filters[group]:
        info_filters[group].add(value)
    btn = tk.Button(frame, text=value, width=14, relief=tk.SUNKEN, command=lambda: toggle_filter(group, value, btn))
    btn.pack(padx=1, pady=1)

def toggle_filter(group, value, button):
    if value in info_filters[group]:
        info_filters[group].remove(value)
        button.config(relief=tk.RAISED)
    else:
        info_filters[group].add(value)
        button.config(relief=tk.SUNKEN)
    update_line_opacity()

# Main plot

def plot_data(new=False):
    global fig, ax, line_refs
    feats = [feature_listbox.get(i) for i in feature_listbox.curselection()]
  
    seqs = [sequence_listbox.get(i) for i in sequence_listbox.curselection()]
    labels = [label_listbox.get(i) for i in label_listbox.curselection()]
    nan_keys = [nan_methods[nan_listbox.get(i)] for i in nan_listbox.curselection()]
    tx_keys = [transform_methods[transform_listbox.get(i)] for i in transform_listbox.curselection()]

    if not feats or not seqs or not labels:
        print("❗ Select at least one feature, sequence, and label.")
        return
    print(fig)
    if new or fig is None:
        fig, ax = plt.subplots(figsize=(14, 6))
        line_refs.clear()
        for frame in [frame_info_feature, frame_info_seq, frame_info_label, frame_info_nan, frame_info_tx]:
            for widget in frame.winfo_children(): widget.destroy()
        for group in info_filters: info_filters[group].clear()

    added = {"Feature": set(), "Sequence": set(), "Label": set(), "NaN": set(), "Transform": set()}

    for label in labels:
        label_color = colors[unique_labels.index(label) % len(colors)]
        if label not in added['Label']:
            add_toggle_button(frame_info_label, "Label", label)
            added['Label'].add(label)
        for seq in seqs:
            if str(seq) not in added['Sequence']:
                add_toggle_button(frame_info_seq, "Sequence", str(seq))
                added['Sequence'].add(str(seq))
            base = raw_df[(raw_df['Label'] == label) & (raw_df['sequence'] == int(seq))].sort_values('timestep')
            if base.empty: continue
            for nan in nan_keys:
                if nan not in added['NaN']:
                    add_toggle_button(frame_info_nan, "NaN", nan)
                    added['NaN'].add(nan)
                df_nan = apply_nan(base.copy(), nan)
                for tx in tx_keys or ['']:
                    txkey = tx if tx else 'raw'
                    if txkey not in added['Transform']:
                        add_toggle_button(frame_info_tx, "Transform", txkey)
                        added['Transform'].add(txkey)
                    df_tx = apply_transform(df_nan.copy(), tx) if tx else df_nan.copy()
                    for feat in feats:
                        if feat not in added['Feature']:
                            add_toggle_button(frame_info_feature, "Feature", feat)
                            added['Feature'].add(feat)
                        y = df_tx[feat].values
                        x = df_tx['timestep'].values
                        mask = ~np.isnan(y)
                        y = y[mask]
                        x = x[mask]
                        if trim_var.get().isdigit():
                            x, y = trim_timestep(x, y, int(trim_var.get()))
                        elif reset_timestep.get():
                            x = np.arange(1, len(y)+1)
                        if len(x) == 0 or len(y) == 0: continue
                        marker = markers[feature_cols.index(feat) % len(markers)]
                        line = ax.plot(x, y, label=f"{feat}|{label}|{seq}|{nan}|{txkey}",
                                       color=label_color, alpha=1.0, marker=marker, markersize=4, linestyle='-')
                        line_refs.append((line[0], (feat, label, str(seq), nan, txkey)))

    ax.set_title("Interactive Time Series Filter")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Reset plot
def reset_plot():
    global fig, ax, line_refs
    if ax:
        ax.clear()
        fig.canvas.draw()
        line_refs.clear()
        for frame in [frame_info_feature, frame_info_seq, frame_info_label, frame_info_nan, frame_info_tx]:
            for widget in frame.winfo_children(): widget.destroy()
        for group in info_filters: info_filters[group].clear()

# Buttons
ttk.Button(frame_mid, text="Plot", command=lambda: plot_data(new=False)).pack(fill='x', pady=5)
ttk.Button(frame_mid, text="New Plot", command=lambda: plot_data(new=True)).pack(fill='x', pady=5)
ttk.Button(frame_mid, text="Reset", command=reset_plot).pack(fill='x', pady=5)

# Run
root.mainloop()
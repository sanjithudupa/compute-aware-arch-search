import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(project_root, "evaluation", "timing_results.csv")

df = pd.read_csv(results_path)
df = df[df['invalidated'] == False]

# Get most recent run for each config
df['test_start_timestamp'] = pd.to_datetime(df['test_start_timestamp'])
most_recent = df.groupby('config_name')['test_start_timestamp'].max().reset_index()
df_recent = df.merge(most_recent, on=['config_name', 'test_start_timestamp'])

# Get unique configs and context lengths
configs = sorted(df_recent['config_name'].unique())
context_lengths = sorted(df_recent['context_length'].unique())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
context_colors = {ctx: colors[i % len(colors)] for i, ctx in enumerate(context_lengths)}

# Calculate throughput
df_recent['throughput'] = df_recent['n_generated'] / df_recent['time']

# Figure 1: Time To First Token
# First, collect raw values and normalize within each sequence length group
raw_values_dict = {}
normalized_values_dict = {}
min_max_dict = {}

for ctx_len in context_lengths:
    raw_values = []
    for config in configs:
        row = df_recent[(df_recent['config_name'] == config) & (df_recent['context_length'] == ctx_len)]
        if len(row) > 0:
            raw_values.append(row['ttft'].values[0])
        else:
            raw_values.append(0)
    
    raw_values_dict[ctx_len] = raw_values
    # Get min/max for this sequence length group (across all configs)
    non_zero_vals = [v for v in raw_values if v > 0]
    if non_zero_vals:
        min_val = min(non_zero_vals)
        max_val = max(raw_values)
    else:
        min_val = 0
        max_val = 0
    
    # Normalize to 0.1-1.0 range within this sequence length group (so min bars are still visible)
    if max_val > min_val:
        normalized = [0.1 + 0.9 * (v - min_val) / (max_val - min_val) if v > 0 else 0.05 for v in raw_values]
    elif max_val == min_val and max_val > 0:
        normalized = [0.5] * len(raw_values)
    else:
        normalized = [0.05] * len(raw_values)
    
    normalized_values_dict[ctx_len] = normalized
    min_max_dict[ctx_len] = (min_val, max_val)

fig1, ax1 = plt.subplots(figsize=(16, 8))
fig1.subplots_adjust(left=0.15)  # Make room for left axes
x = np.arange(len(configs))
width = 0.25

bars_list = []
for i, ctx_len in enumerate(context_lengths):
    normalized = normalized_values_dict[ctx_len]
    offset = (i - len(context_lengths)/2 + 0.5) * width
    bars = ax1.bar(x + offset, normalized, width, label=f'Sequence Length {ctx_len}', 
                   color=context_colors[ctx_len], edgecolor='black', linewidth=0.5)
    bars_list.append(bars)

ax1.set_ylim(0, 1.15)
ax1.set_xlabel('Model Config', fontsize=12, fontweight='bold')
ax1.set_title('Time To First Token by Model Config and Sequence Length', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=10)
ax1.legend(title='Sequence Length', fontsize=10, title_fontsize=11, loc='upper right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# Hide main y-axis ticks
ax1.set_yticks([])
ax1.spines['left'].set_visible(False)

# Create a single combined y-axis on the LEFT with black ticks
base_offset = 50
ax_combined = ax1.twinx()
ax_combined.spines['left'].set_position(('outward', base_offset))
ax_combined.spines['left'].set_color('black')
ax_combined.spines['right'].set_visible(False)
ax_combined.set_ylim(0, 1.15)
# Make sure the left spine extends fully from bottom to top
ax_combined.spines['left'].set_bounds(0, 1.15)

# Make top and bottom spines on combined axis extend from left spine to right edge
ax_combined.spines['top'].set_visible(True)
ax_combined.spines['bottom'].set_visible(True)
ax_combined.spines['top'].set_color('black')
ax_combined.spines['bottom'].set_color('black')


# Keep main axis spines visible
ax1.spines['top'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['top'].set_color('black')
ax1.spines['bottom'].set_color('black')
ax_combined.tick_params(axis='y', colors='black', labelsize=0, labelleft=True, labelright=False, pad=0, width=1)
ax_combined.yaxis.set_label_position('left')
ax_combined.yaxis.set_ticks_position('left')
ax_combined.set_ylabel('TTFT (seconds)', fontsize=11, fontweight='bold', rotation=90, labelpad=15)


# Set tick positions in normalized space
tick_positions = [0.1, 0.5, 1.0]
ax_combined.set_yticks(tick_positions)
ax_combined.set_yticklabels([''] * len(tick_positions))  # Hide default labels

# Manually add colored text labels stacked vertically next to each tick
x_label_offset = -0.02  # Distance from axis line
label_spacing = 0.04  # Vertical spacing between stacked labels

for pos in tick_positions:
    values = []
    for ctx_len in context_lengths:
        min_val, max_val = min_max_dict[ctx_len]
        # Map normalized position back to actual value
        # pos 0.1 maps to min_val, pos 1.0 maps to max_val
        if max_val > min_val:
            actual_val = min_val + (pos - 0.1) / 0.9 * (max_val - min_val)
        else:
            actual_val = min_val
        values.append((actual_val, ctx_len))
    
    # Stack labels vertically, starting from the tick position
    y_offset = label_spacing * (len(values) - 1) / 2  # Center the stack
    
    for i, (val, ctx_len) in enumerate(values):
        y_pos = pos - y_offset + i * label_spacing
        # Add the number with its color, stacked vertically
        ax1.text(x_label_offset, y_pos, f'{val:.2f}', transform=ax1.get_yaxis_transform(), 
                color=context_colors[ctx_len], fontsize=9, ha='right', va='center', fontweight='bold')

# Add actual value labels on bars
for i, (bars, ctx_len) in enumerate(zip(bars_list, context_lengths)):
    raw_vals = raw_values_dict[ctx_len]
    for bar, raw_val in zip(bars, raw_vals):
        height = bar.get_height()
        if raw_val > 0 or height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{raw_val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()

# Draw connector lines AFTER tight_layout using renderer to get exact positions
fig1.canvas.draw()
renderer = fig1.canvas.get_renderer()

# Get exact pixel positions
bbox_ax = ax1.get_window_extent(renderer)
bbox_spine = ax_combined.spines['left'].get_window_extent(renderer)

# Convert to figure coordinates
fig_width, fig_height = fig1.get_size_inches() * fig1.dpi
x_left_plot = bbox_ax.x0 / fig_width
x_left_spine = bbox_spine.x0 / fig_width

# Create transform: x in figure coords, y in data coords
from matplotlib.transforms import blended_transform_factory
trans_conn = blended_transform_factory(fig1.transFigure, ax1.transData)

# Draw horizontal connector lines
from matplotlib.lines import Line2D
conn_bottom = Line2D([x_left_spine, x_left_plot], [0, 0], 
                     transform=trans_conn, color='black', linewidth=1.5, 
                     clip_on=False, zorder=10)
conn_top = Line2D([x_left_spine, x_left_plot], [1.15, 1.15],
                  transform=trans_conn, color='black', linewidth=1.5,
                  clip_on=False, zorder=10)
ax1.add_line(conn_bottom)
ax1.add_line(conn_top)

output_path = os.path.join(project_root, 'evaluation', 'ttft_chart.png')
fig1.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved TTFT chart to {output_path}")
plt.close(fig1)

# Figure 2: Model Token Throughput
fig2, ax2 = plt.subplots(figsize=(14, 7))

bars_list2 = []
for i, ctx_len in enumerate(context_lengths):
    values = []
    for config in configs:
        row = df_recent[(df_recent['config_name'] == config) & (df_recent['context_length'] == ctx_len)]
        if len(row) > 0:
            values.append(row['throughput'].values[0])
        else:
            values.append(0)
    
    offset = (i - len(context_lengths)/2 + 0.5) * width
    bars = ax2.bar(x + offset, values, width, label=f'Sequence Length {ctx_len}', 
                   color=context_colors[ctx_len], edgecolor='black', linewidth=0.5)
    bars_list2.append(bars)

ax2.set_xlabel('Model Config', fontsize=12, fontweight='bold')
ax2.set_ylabel('Token Throughput (tok/sec)', fontsize=12, fontweight='bold')
ax2.set_title('Model Token Throughput by Model Config and Sequence Length', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=10)
ax2.legend(title='Sequence Length', fontsize=10, title_fontsize=11)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# Add value labels on bars
for bars in bars_list2:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
output_path2 = os.path.join(project_root, 'evaluation', 'throughput_chart.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Saved Throughput chart to {output_path2}")

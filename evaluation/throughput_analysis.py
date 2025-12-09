import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(project_root, "evaluation", "timing_results.csv")

df = pd.read_csv(results_path)
df = df[df['invalidated'] == False]

df['test_start_timestamp'] = pd.to_datetime(df['test_start_timestamp'])
most_recent = df.groupby('config_name')['test_start_timestamp'].max().reset_index()
df_recent = df.merge(most_recent, on=['config_name', 'test_start_timestamp'])

configs = sorted(df_recent['config_name'].unique())
context_lengths = sorted(df_recent['context_length'].unique())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
context_colors = {ctx: colors[i % len(colors)] for i, ctx in enumerate(context_lengths)}

df_recent['throughput'] = df_recent['n_generated'] / df_recent['time']

print("\n" + "="*80)
print("PERFORMANCE METRICS SUMMARY")
print("="*80)

for ctx_len in context_lengths:
    print(f"\nContext Length: {ctx_len}")
    print("-" * 80)
    print(f"{'Config':<15} {'Throughput (tok/s)':>18}  {'TTFT (s)':>13}  {'Throughput Speedup':>18}  {'TTFT Speedup':>13}")
    print("-" * 80)
    
    control_row = df_recent[(df_recent['config_name'] == 'control') & (df_recent['context_length'] == ctx_len)]
    if len(control_row) == 0:
        print(f"Warning: No baseline (control) data for context_length={ctx_len}")
        continue
    
    control_throughput = control_row['throughput'].values[0]
    control_ttft = control_row['ttft'].values[0]
    
    print(f"{'control':<15} {control_throughput:>18.2f}  {control_ttft:>13.4f}  {1.00:>18.2f}x  {1.00:>13.2f}x")
    
    for config in configs:
        if config == 'control':
            continue
        
        row = df_recent[(df_recent['config_name'] == config) & (df_recent['context_length'] == ctx_len)]
        if len(row) == 0:
            print(f"{config:<15} {'N/A':>18}  {'N/A':>13}  {'N/A':>18}  {'N/A':>13}")
            continue
        
        config_throughput = row['throughput'].values[0]
        config_ttft = row['ttft'].values[0]
        

        if config_throughput > 0:
            throughput_speedup = config_throughput / control_throughput
        else:
            throughput_speedup = 0.0
        
        if config_ttft > 0:
            ttft_speedup = control_ttft / config_ttft
        else:
            ttft_speedup = 0.0
        
        print(f"{config:<15} {config_throughput:>18.2f}  {config_ttft:>13.4f}  {throughput_speedup:>18.2f}x  {ttft_speedup:>13.2f}x")

print("\n" + "="*80 + "\n")

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

    non_zero_vals = [v for v in raw_values if v > 0]
    if non_zero_vals:
        min_val = min(non_zero_vals)
        max_val = max(raw_values)
    else:
        min_val = 0
        max_val = 0
    
    if max_val > min_val:
        normalized = [0.1 + 0.9 * (v - min_val) / (max_val - min_val) if v > 0 else 0.05 for v in raw_values]
    elif max_val == min_val and max_val > 0:
        normalized = [0.5] * len(raw_values)
    else:
        normalized = [0.05] * len(raw_values)
    
    normalized_values_dict[ctx_len] = normalized
    min_max_dict[ctx_len] = (min_val, max_val)

fig1, ax1 = plt.subplots(figsize=(16, 8))
fig1.subplots_adjust(left=0.15) 
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

ax1.set_yticks([])
ax1.spines['left'].set_visible(False)

base_offset = 50
ax_combined = ax1.twinx()
ax_combined.spines['left'].set_position(('outward', base_offset))
ax_combined.spines['left'].set_color('black')
ax_combined.spines['right'].set_visible(False)
ax_combined.set_ylim(0, 1.15)

ax_combined.spines['left'].set_bounds(0, 1.15)

ax_combined.spines['top'].set_visible(True)
ax_combined.spines['bottom'].set_visible(True)
ax_combined.spines['top'].set_color('black')
ax_combined.spines['bottom'].set_color('black')


ax1.spines['top'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['top'].set_color('black')
ax1.spines['bottom'].set_color('black')
ax_combined.tick_params(axis='y', colors='black', labelsize=0, labelleft=True, labelright=False, pad=0, width=1)
ax_combined.yaxis.set_label_position('left')
ax_combined.yaxis.set_ticks_position('left')
ax_combined.set_ylabel('TTFT (seconds)', fontsize=11, fontweight='bold', rotation=90, labelpad=15)


tick_positions = [0.1, 0.5, 1.0]
ax_combined.set_yticks(tick_positions)
ax_combined.set_yticklabels([''] * len(tick_positions)) 

x_label_offset = -0.02 
label_spacing = 0.04 

for pos in tick_positions:
    values = []
    for ctx_len in context_lengths:
        min_val, max_val = min_max_dict[ctx_len]
        if max_val > min_val:
            actual_val = min_val + (pos - 0.1) / 0.9 * (max_val - min_val)
        else:
            actual_val = min_val
        values.append((actual_val, ctx_len))
    
    y_offset = label_spacing * (len(values) - 1) / 2 
    
    for i, (val, ctx_len) in enumerate(values):
        y_pos = pos - y_offset + i * label_spacing
        ax1.text(x_label_offset, y_pos, f'{val:.2f}', transform=ax1.get_yaxis_transform(), 
                color=context_colors[ctx_len], fontsize=9, ha='right', va='center', fontweight='bold')

for i, (bars, ctx_len) in enumerate(zip(bars_list, context_lengths)):
    raw_vals = raw_values_dict[ctx_len]
    for bar, raw_val in zip(bars, raw_vals):
        height = bar.get_height()
        if raw_val > 0 or height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{raw_val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()

fig1.canvas.draw()
renderer = fig1.canvas.get_renderer()

bbox_ax = ax1.get_window_extent(renderer)
bbox_spine = ax_combined.spines['left'].get_window_extent(renderer)

fig_width, fig_height = fig1.get_size_inches() * fig1.dpi
x_left_plot = bbox_ax.x0 / fig_width
x_left_spine = bbox_spine.x0 / fig_width

from matplotlib.transforms import blended_transform_factory
trans_conn = blended_transform_factory(fig1.transFigure, ax1.transData)

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

fig1_log, ax1_log = plt.subplots(figsize=(14, 7))
x = np.arange(len(configs))
width = 0.25

bars_list_log = []
for i, ctx_len in enumerate(context_lengths):
    values = []
    for config in configs:
        row = df_recent[(df_recent['config_name'] == config) & (df_recent['context_length'] == ctx_len)]
        if len(row) > 0:
            values.append(row['ttft'].values[0])
        else:
            values.append(0)
    
    offset = (i - len(context_lengths)/2 + 0.5) * width
    bars = ax1_log.bar(x + offset, values, width, label=f'Sequence Length {ctx_len}', 
                       color=context_colors[ctx_len], edgecolor='black', linewidth=0.5)
    bars_list_log.append(bars)

ax1_log.set_yscale('log')
ax1_log.set_xlabel('Model Config', fontsize=12, fontweight='bold')
ax1_log.set_ylabel('Time To First Token (seconds, log scale)', fontsize=12, fontweight='bold')
ax1_log.set_title('Time To First Token by Model Config and Sequence Length', fontsize=14, fontweight='bold')
ax1_log.set_xticks(x)
ax1_log.set_xticklabels(configs, fontsize=10)
ax1_log.legend(title='Sequence Length', fontsize=10, title_fontsize=11, loc='upper right')
ax1_log.grid(axis='y', alpha=0.3, linestyle='--')
ax1_log.set_axisbelow(True)

for bars in bars_list_log:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1_log.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
output_path_log = os.path.join(project_root, 'evaluation', 'ttft_chart_log.png')
fig1_log.savefig(output_path_log, dpi=300, bbox_inches='tight')
print(f"Saved TTFT log scale chart to {output_path_log}")
plt.close(fig1_log)

# FIGURE 2

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

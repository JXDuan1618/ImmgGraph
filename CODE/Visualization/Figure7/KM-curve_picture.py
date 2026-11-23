# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

# 与原脚本一致：字体设置为 Arial
mpl.rcParams['font.family'] = 'Arial'  # 原脚本同样设置了字体【turn26file0†L49-L62】

# 读取你已经分好组的数据表（路径按需改成你的本地路径）
train_df = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_5\KM_curve\new_20260921\picture_and_groupcsv\group_csv\majority_train_PFS_risk.csv')
val_df   = pd.read_csv(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_5\KM_curve\new_20260921\picture_and_groupcsv\group_csv\majority_valid_PFS_risk.csv')

# ——与原脚本保持等价：提取生存时间与事件列（列名与你文件相同）【turn26file0†L103-L118】
train_overall_survival = train_df['overall_survival']
train_vital_status     = train_df['vital_status']
val_overall_survival   = val_df['overall_survival']
val_vital_status       = val_df['vital_status']

# 直接用你表里的分组（保持与原脚本后续使用的布尔掩码一致的语义）
train_high_risk = (train_df['risk_group'] == 'High-risk')
train_low_risk  = (train_df['risk_group'] == 'Low-risk')
val_high_risk   = (val_df['risk_group'] == 'High-risk')
val_low_risk    = (val_df['risk_group'] == 'Low-risk')

# 拟合 KM（与原脚本相同的四个拟合器与 fit 顺序）【turn26file0†L132-L146】
kmf_train_high = KaplanMeierFitter()
kmf_train_low  = KaplanMeierFitter()
kmf_val_high   = KaplanMeierFitter()
kmf_val_low    = KaplanMeierFitter()

kmf_train_high.fit(train_overall_survival[train_high_risk], train_vital_status[train_high_risk])
kmf_train_low.fit (train_overall_survival[train_low_risk],  train_vital_status[train_low_risk])
kmf_val_high.fit  (val_overall_survival[val_high_risk],     val_vital_status[val_high_risk])
kmf_val_low.fit   (val_overall_survival[val_low_risk],      val_vital_status[val_low_risk])

# 与原脚本一致的 logrank 检验与 p 值下限处理【turn26file0†L148-L163】
resultt   = logrank_test(train_overall_survival[train_high_risk], train_overall_survival[train_low_risk],
                         train_vital_status[train_high_risk],      train_vital_status[train_low_risk])
p_valuett = resultt.p_value
if p_valuett < 0.0001:
    p_valuett = 0.0001

results   = logrank_test(val_overall_survival[val_high_risk], val_overall_survival[val_low_risk],
                         val_vital_status[val_high_risk],      val_vital_status[val_low_risk])
p_valuets = results.p_value
if p_valuets < 0.0001:
    p_valuets = 0.0001

# 与原脚本一致：强制验证集曲线首点为 1.0（如果需要）【turn26file0†L164-L169】
kmf_val_high.survival_function_.iloc[0, 0] = 1.0
kmf_val_low.survival_function_.iloc[0, 0]  = 1.0

# ===== 训练集图（完全复刻你的绘图参数）=====【turn26file0†L170-L206】
fig, ax1 = plt.subplots(figsize=(12, 8))

kmf_train_high.plot_survival_function(
    ax=ax1, label='High-risk Group',
    ci_show=True, ci_alpha=0.2, show_censors=True,
    censor_styles={'marker': '|', 'ms': 12, 'mew': 3},
    linewidth=6, color="#4d92c3", fontsize=55
)
kmf_train_low.plot_survival_function(
    ax=ax1, label='Low-risk Group',
    ci_show=True, ci_alpha=0.2, show_censors=True,
    censor_styles={'marker': '|', 'ms': 12, 'mew': 3},
    linewidth=6, color="#ff9e4a", fontsize=55
)

add_at_risk_counts(
    kmf_train_high, kmf_train_low, ax=ax1,
    rows_to_show=['At risk'], labels=['High Risk', 'Low Risk'],
    fontsize=60,
    at_risk_count_from_start_of_period=True
)

ax1.set_xlabel('Time (Months)', fontsize=60, labelpad=150)
ax1.set_ylabel('Survival Probability', fontsize=60, labelpad=30)
ax1.set_title('Majority Voting', fontsize=60, pad=30)
ax1.annotate(
    f'p= {p_valuett:.4f}', xy=(0.5, 0.8), xycoords='axes fraction',
    fontsize=60,
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.3')
)

for spine in ax1.spines.values():
    spine.set_linewidth(6)
ax1.tick_params(axis='both', which='major', labelsize=60, length=15, width=6)
ax1.legend().set_visible(False)
plt.subplots_adjust(bottom=0.1)
plt.savefig(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_5\KM_curve\new_20260921\picture_and_groupcsv\picture\majority_train.png', bbox_inches='tight')

# ===== 验证集图（完全复刻你的绘图参数）=====【turn26file0†L208-L225】
fig, ax2 = plt.subplots(figsize=(12, 8))

kmf_val_high.plot_survival_function(
    ax=ax2, label='High-risk Group',
    ci_show=True, ci_alpha=0.2, show_censors=True,
    censor_styles={'marker': '|', 'ms': 12, 'mew': 3},
    linewidth=6, color="#4d92c3", fontsize=55
)
kmf_val_low.plot_survival_function(
    ax=ax2, label='Low-risk Group',
    ci_show=True, ci_alpha=0.2, show_censors=True,
    censor_styles={'marker': '|', 'ms': 12, 'mew': 3},
    linewidth=6, color="#ff9e4a", fontsize=55
)

add_at_risk_counts(
    kmf_val_high, kmf_val_low, ax=ax2,
    rows_to_show=['At risk'], labels=['High Risk', 'Low Risk'],
    fontsize=60,
    at_risk_count_from_start_of_period=True
)

ax2.set_xlabel('Time (Months)', fontsize=60, labelpad=150)
ax2.set_ylabel('Survival Probability', fontsize=60, labelpad=30)
ax2.set_title('Majority Voting', fontsize=60, pad=30)
ax2.annotate(
    f'p= {p_valuets:.4f}', xy=(0.5, 0.8), xycoords='axes fraction',
    fontsize=60,
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.3')
)

for spine in ax2.spines.values():
    spine.set_linewidth(6)
ax2.tick_params(axis='both', which='major', labelsize=60, length=15, width=6)
ax2.legend().set_visible(False)
#ax2.legend(loc='upper left', fontsize=60, bbox_to_anchor=(1, 1))
plt.subplots_adjust(bottom=0.1)
plt.savefig(r'E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_5\KM_curve\new_20260921\picture_and_groupcsv\picture\majority_valid.png', bbox_inches='tight')

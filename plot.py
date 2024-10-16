#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# 中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.sans-serif'] = ['Heiti TC']

# Colors
COLORS = ['#89c3eb', '#90e667', '#e0b5d3', '#be1e3e']


# 数据
languages = ['ar', 'my', 'zh', 'en', 'fr', 'hi', 'ja', 'ko', 'ru', 'es', 'tr']
# mbart, ROUGE-1
# mbart_finetune = [33.98, 43.25, 37.16, 40.14, 33.29, 38.90, 46.01, 20.51, 32.74, 32.57, 32.19]
# mbart_prefix = [31.95, 41.27, 33.67, 37.23, 30.98, 37.40, 44.66, 20.14, 31.16, 30.23, 28.11]
# mbart_lora = [30.31, 37.08, 32.04, 35.17, 27.99, 35.01, 39.53, 12.20, 25.66, 29.58, 27.55]
# mbart, ROUGE-L
mbart_finetune = [27.51, 31.74, 30.28, 31.73, 25.16, 31.32, 34.24, 19.07, 25.36, 23.72, 28.38]
mbart_prefix = [26.30, 30.46, 27.48, 29.27, 24.95, 30.19, 33.25, 18.90, 24.66, 21.59, 24.93]
mbart_lora = [25.17, 27.23, 26.14, 27.35, 22.35, 27.97, 28.52, 11.17, 20.19, 21.60, 24.11]

# 设置柱子的位置
total_width, n = 0.72, 3
width = total_width / n
x = np.arange(len(languages))
x = x - (total_width - width) / 2

# 绘制多系列柱状图
plt.figure(figsize=(8, 4))
plt.bar(x, mbart_finetune, width, label='FT', color=COLORS[0])
plt.bar(x+width, mbart_prefix, width, label='PT', color=COLORS[1])
plt.bar(x+width*2, mbart_lora, width, label='LoRA', color=COLORS[2])
plt.ylabel("ROUGE-L", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 添加标题和标签
# plt.title('ROUGE-L scores of mBART$_\mathdefault{LARGE}$', fontsize=18)
# plt.xlabel('')
# plt.ylabel('')
plt.tick_params(labelsize=11)
plt.xticks(x+width, languages, fontsize=14)
plt.legend(fontsize=11)  # 显示图例

plt.savefig("fig/mbart-result.png")
plt.show()

# mt5, ROUGE-L
mt5_finetune = [29.16, 33.15, 33.41, 29.88, 28.20, 32.01, 37.36, 22.36, 26.17, 24.07, 29.26]
mt5_prefix = [25.58, 30.59, 28.21, 28.02, 24.91, 29.27, 32.67, 16.47, 22.55, 21.85, 21.73]
mt5_lora = [23.12, 29.13, 26.33, 25.90, 22.68, 27.60, 29.58, 12.50, 20.41, 20.89, 22.42]

plt.figure(figsize=(8, 4))
plt.bar(x, mt5_finetune, width, label='FT', color=COLORS[0])
plt.bar(x+width, mt5_prefix, width, label='PT', color=COLORS[1])
plt.bar(x+width*2, mt5_lora, width, label='LoRA', color=COLORS[2])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tick_params(labelsize=11)
plt.xticks(x+width, languages, fontsize=14)
plt.ylabel("ROUGE-L", fontsize=12)
plt.legend(fontsize=11)
plt.savefig("fig/mt5-result.png")
plt.show()

# mbart, optimal length
mbart_prefix_optlen = [26.56, 30.46, 27.98, 29.27, 25.15, 30.53, 33.25, 18.90, 24.66, 23.19, 25.79]
mbart_prefix_optlen_len = [30, 100, 50, 100, 300, 300, 100, 100, 100, 200, 200]

total_width, n = 0.72, 2
width = total_width / n
x = np.arange(len(languages))
x = x - (total_width - width) / 2

# plt.figure(figsize=(8, 4))
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.bar(x, mbart_finetune, width, label='FT', color=COLORS[0])
ax1.bar(x+width, mbart_prefix_optlen, width, label='PT', color=COLORS[1])
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.tick_params(labelsize=11)
plt.xticks(x+width/2, languages, fontsize=14)
plt.ylabel("ROUGE-L", fontsize=12)
plt.legend(fontsize=11)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
ax2.set_ylabel('optimal prefix length', color=COLORS[3], fontsize=12)
ax2.scatter(x+width/2, mbart_prefix_optlen_len, marker="D", color=COLORS[3])
plt.savefig("fig/mbart-opt-len.png")
plt.show()

# mbart, optimal length, few-shot
mbart_finetune_fewshot = [23.01, 28.83, 23.19, 23.06, 23.71, 24.25, 34.33, 16.47, 19.84, 20.74, 21.48]
mbart_prefix_optlen_fewshot = [22.67, 30.72, 23.29, 23.95, 23.88, 26.38, 32.52, 18.54, 21.51, 21.38, 21.67]

plt.figure(figsize=(8, 4))
plt.bar(x, mbart_finetune_fewshot, width, label='FT', color=COLORS[0])
plt.bar(x+width, mbart_prefix_optlen_fewshot, width, label='PT', color=COLORS[1])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tick_params(labelsize=11)
plt.xticks(x+width/2, languages, fontsize=14)
plt.ylabel("ROUGE-L", fontsize=12)
plt.legend(fontsize=11)
plt.savefig("fig/mbart-few-shot.png")
plt.show()
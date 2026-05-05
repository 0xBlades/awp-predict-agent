#!/usr/bin/env python3
"""Generate AWP Predict daily report for Telegram delivery."""
import os
import sys

os.environ['AGENT_HOME'] = '/home/ubuntu/.awp-predict-main'
sys.path.insert(0, '/home/ubuntu/.hermes/skills/predict-agent')

import memory_manager as mem_mod
from datetime import datetime

wr = mem_mod.get_winrate()
top = mem_mod.get_top_markets(3)
heatmap = mem_mod.load_heatmap()

report = f"📊 AWP Predict Daily Report\n"
report += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
report += f"━━━━━━━━━━━━━━━━━━━━━\n"
report += f"Win Rate: {wr['rate']}% ({wr['wins']}W / {wr['losses']}L / {wr['total']} total)\n"

if top:
    report += f"\n🏆 Top Markets:\n"
    for m in top:
        report += f"  {m['token']}: {m['rate']*100:.1f}% ({m['total']} trades)\n"

if heatmap:
    report += f"\n📈 All Assets:\n"
    for token, stats in sorted(heatmap.items()):
        total = stats.get('wins', 0) + stats.get('losses', 0)
        if total > 0:
            rate = stats.get('wins', 0) / total * 100
            emoji = '🟢' if rate >= 50 else '🔴'
            report += f"  {emoji} {token}: {rate:.0f}% ({stats.get('wins',0)}W/{stats.get('losses',0)}L)\n"

# Log snapshot
mem_mod.log_winrate_snapshot()

print(report)

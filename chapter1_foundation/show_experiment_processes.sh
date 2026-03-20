#!/bin/bash
# 在终端显示 6 个实验训练主进程 (v3)
echo ""
echo "========== 实验 v3 训练主进程 (6 个 seed) =========="
echo ""
printf "  %-8s %-14s %-6s  %s\n" "PID" "运行时长" "CPU%" "Seed"
printf "  %s\n" "--------------------------------------------------------"
ps -eo pid,etime,pcpu,args --no-headers 2>/dev/null | grep "run_experiment" | grep "experiment_results_v3" | grep "python3 -m" | grep -v "bin/bash" | \
  awk '{
    args=""; for(i=4;i<=NF;i++) args=args $i " ";
    if(match(args,/seed_[0-9]+/)) seed=substr(args,RSTART,RLENGTH);
    key=seed;
    if(key=="") next;
    if(!(key in seen) || length($2)>length(etime[key])) { pid[key]=$1; etime[key]=$2; cpu[key]=$3; seen[key]=1; }
  }
  END{
    n=split("seed_42 seed_153 seed_264 seed_375 seed_486 seed_597", s);
    for(i=1;i<=n;i++) if(s[i] in pid) printf "  %-8s %-14s %-6s  %s\n", pid[s[i]], etime[s[i]], cpu[s[i]]"%", s[i];
  }'
echo ""
echo "运行时长 12:49:xx = 约 12 小时 49 分。"
echo ""
echo "备用（在跑训练的机器上直接执行）："
echo "  ps -eo pid,etime,pcpu,args | grep run_experiment | grep experiment_results_v3 | grep -v grep"
echo ""

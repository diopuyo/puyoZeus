#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
echo "video / boundaries / winners / 試合数 / 学習可能"
echo "----------------------------------------------------"
total_matches=0
trainable=0
for v in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19; do
    vp="data/frames/video_${v}.mp4"
    bp_v5="data/verify/match_boundaries_v5/video_${v}/matches.tsv"
    bp_v4="data/verify/match_boundaries_v4/video_${v}/matches.tsv"
    wp="data/verify/match_winners_v${v}.tsv"
    v_ok="-"
    [ -f "$vp" ] && v_ok="Y"
    n_match=0
    b_kind="-"
    if [ -f "$bp_v5" ]; then
        n_match=$(awk 'NR>1' "$bp_v5" | wc -l)
        b_kind="v5"
    elif [ -f "$bp_v4" ]; then
        n_match=$(awk 'NR>1' "$bp_v4" | wc -l)
        b_kind="v4"
    fi
    n_win=0
    w_ok="-"
    if [ -f "$wp" ]; then
        n_win=$(awk 'NR>1 && $4 ~ /^[12]P$/' "$wp" | wc -l)
        w_ok="Y"
    fi
    usable=$(( n_win < n_match ? n_win : n_match ))
    can_train="-"
    if [ "$v_ok" = "Y" ] && [ "$n_win" -gt 0 ]; then
        can_train="Y"
        trainable=$((trainable + 1))
        total_matches=$((total_matches + usable))
    fi
    printf "v%s  vid=%s  bnd=%s(%3d 試合)  win=%s(%3d 件)  学習=%s\n" \
        "$v" "$v_ok" "$b_kind" "$n_match" "$w_ok" "$n_win" "$can_train"
done
echo "----------------------------------------------------"
echo "学習可能動画数: $trainable / 19"
echo "学習可能試合数: $total_matches"

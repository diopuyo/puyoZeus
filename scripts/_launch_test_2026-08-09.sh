#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
setsid -f bash -c 'echo start_$(date +%s) > logs/_setsid_test.log; sleep 60; echo end >> logs/_setsid_test.log'
echo probe_rc=$?

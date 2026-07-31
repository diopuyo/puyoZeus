#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
awk 'NR==6{print "c8:",$0}
     NR==13{print "c15:",$0}
     NR==15{print "c17:",$0}
     NR==21{print "c23:",$0}
     NR==29{print "c31:",$0}' data/pl_new_missing.tsv

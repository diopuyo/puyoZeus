#!/bin/bash
pkill -f "scripts.next_shift_m0_cv_20261001 --root" ; sleep 2; pgrep -f "scripts.next_shift_m0_cv_20261001 --root" | wc -l

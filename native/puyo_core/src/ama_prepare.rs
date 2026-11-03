//! 根の準備中だけ相手を進める。後続段の相手は既存の連鎖終了待ちを使う。
use super::*;
use std::time::Instant;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Preparation {
    pub mode: String,
    pub horizon: usize,
    pub compute_ms: f64,
    pub reused: bool,
    pub start_index: usize,
    pub end_index: usize,
    pub fire_ms: Option<u64>,
    pub end_ms: u64,
    pub board: BitBoard,
    pub score: i32,
    pub unsupported: Option<&'static str>,
}

pub(super) fn apply(node: &mut Branch, state: &mut State, initial: &State, tape: &[Pair],
             cfg: &Config, rules: &Rules, opt: &anytime::Options, k: usize,
             shared: &mut HashMap<usize, Outcome>) {
    let cutoff = node.fire_ms.unwrap_or(node.end_ms);
    // 発火済み相手は完走盤面を受け取り、残り演出中は新しい組を置かない。
    let available = cutoff.saturating_sub(opt.opponent_busy_ms);
    let horizon = if opt.preparation_mode == "candidate" {
        (available / rules.hand_ms) as usize
    } else { opt.opponent_horizon.unwrap_or(k) };
    let cacheable = opt.preparation_mode != "separate";
    let reused = cacheable && shared.contains_key(&horizon);
    let began = Instant::now();
    let trace = if reused { shared[&horizon].clone() } else {
        let plan = Plan { policy: Policy::Main, hands: horizon, target: opt.opponent_target };
        ama_f::run(initial.boards[1], &tape[initial.cursors[1]..], plan, cfg, 0, true)
    };
    if cacheable && !reused { shared.insert(horizon, trace.clone()); }
    let compute_ms = began.elapsed().as_secs_f64() * 1000.0;
    let count = ((available / rules.hand_ms) as usize).min(trace.steps.len());
    let snapshot = trace.steps.get(count.wrapping_sub(1));
    let fire = snapshot.filter(|s| s.score > 0)
        .map(|s| opt.opponent_busy_ms + s.hand as u64 * rules.hand_ms);
    let end = fire.map(|t| t + rules.chain_ms[snapshot.unwrap().chain as usize])
        .unwrap_or(cutoff.max(opt.opponent_busy_ms));
    let mut prep = Preparation { mode: opt.preparation_mode.clone(), horizon, compute_ms, reused,
        start_index: initial.cursors[1], end_index: initial.cursors[1] + count,
        fire_ms: fire, end_ms: end, board: snapshot.map_or(initial.boards[1], |s| s.board),
        score: if fire.is_some() { trace.total_score } else { 0 }, unsupported: None };
    if fire.is_some() && end < cutoff { prep.unsupported = Some("opponent_landed_before_root_fire"); }
    if fire.is_some() && end >= cutoff && node.fire_ms.is_none() && trace.excess > 0 {
        prep.unsupported = Some("opponent_attack_outlives_no_fire_horizon");
    }
    if fire.is_none() && count < (available / rules.hand_ms) as usize {
        prep.unsupported = Some(if trace.interrupted { "preparation_budget" }
            else if !trace.coverage { "preparation_tape_shortage" } else { "preparation_horizon" });
    }
    if trace.interrupted && prep.unsupported.is_some() { node.outcome.interrupted = true; }
    state.boards[1] = prep.board;
    state.cursors[1] = prep.end_index;
    state.ready[1] = end;
    if fire.is_some() && prep.unsupported.is_none() { overlap(node, state, initial, &trace); }
    if prep.unsupported.is_some() { node.truncated = true; node.balance = None; }
    node.preparation = Some(prep);
}

/// 発火済みでまだ連鎖中なら、相手の送信を自分の予告へ加えて同じ70点規則で相殺する。
fn overlap(node: &mut Branch, state: &mut State, initial: &State, opponent: &Outcome) {
    let incoming = initial.pending[0] + opponent.excess;
    let sent = (node.outcome.total_score + node.outcome.carry_in) / OJAMA_RATE;
    node.outcome.cancelled = sent.min(incoming);
    node.outcome.excess = (sent - incoming).max(0);
    node.outcome.pending = (incoming - sent).max(0);
    state.pending = [node.outcome.pending, node.outcome.excess];
    state.carry[1] = opponent.score_remainder;
    node.pending = state.pending;
}

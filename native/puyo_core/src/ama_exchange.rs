//! 本線・対応の候補木。重み付けや勝率化をせず、葉ごとの生の収支を返す。
//! 相手盤面は最初の発火時点のスナップショット（開始前の積みを二重に進めない）。
use crate::ama_f::{self, Outcome, Plan, Policy, MAX_HANDS, MAX_REQUEST_VALUE, OJAMA_RATE};
use crate::ama_search::{Config, Pair};
use crate::bitboard::{BitBoard, BOARD_COLS, MAX_CHAIN_STEPS};

const MAX_TIME_MS: u64 = 120000;
#[path = "ama_anytime.rs"]
pub mod anytime;
#[path = "ama_prepare.rs"]
pub mod prepare;

#[derive(Clone)]
pub struct Rules {
    pub levels: usize,
    pub hand_ms: u64,
    pub chain_ms: Vec<u64>,
    pub main_targets: Vec<i32>,
    pub deadlines: Vec<usize>,
    pub garbage_order: [usize; BOARD_COLS],
}

#[derive(Clone)]
struct State {
    carry: [i32; 2],
    boards: [BitBoard; 2],
    cursors: [usize; 2],
    ready: [u64; 2],
    landed: [i32; 2],
    pending: [i32; 2],
}

#[derive(Clone, Debug)]
pub struct Branch {
    pub completed_ms: Option<f64>,
    pub progress_truncated: Option<bool>,
    pub unavailable: bool,
    pub preparation: Option<prepare::Preparation>,
    pub side: usize,
    pub level: usize,
    pub plan: Plan,
    pub outcome: Outcome,
    pub start_index: usize,
    pub end_index: usize,
    pub start_ms: u64,
    pub fire_ms: Option<u64>,
    pub end_ms: u64,
    pub available_hands: usize,
    pub balance: Option<i32>,
    pub landed: [i32; 2],
    pub pending: [i32; 2],
    pub truncated: bool,
    pub children: Vec<Branch>,
}

pub fn validate(rules: &Rules, indices: [usize; 2], tape: &[Pair]) -> Result<(), String> {
    if !(2..=3).contains(&rules.levels)
        || rules.hand_ms == 0
        || rules.chain_ms.len() != MAX_CHAIN_STEPS + 1
        || rules.chain_ms[0] != 0
        || rules.chain_ms[1..].iter().any(|&t| t == 0 || t > MAX_TIME_MS)
        || rules.hand_ms > MAX_TIME_MS
    {
        return Err("再帰深さ2/3、正の手時間、0連鎖=0を含む20要素の連鎖時間が必要です".into());
    }
    if rules.main_targets.is_empty()
        || rules.main_targets.iter().any(|&s| s <= 0 || s > MAX_REQUEST_VALUE)
        || rules.deadlines.is_empty()
        || rules.deadlines.iter().any(|&k| k == 0 || k > MAX_HANDS)
        || indices.iter().any(|&i| i > tape.len())
    {
        return Err("候補の得点・期限・共通ツモ開始位置が不正です".into());
    }
    let mut order = rules.garbage_order;
    order.sort();
    if order != [0, 1, 2, 3, 4, 5] {
        return Err("落下順は0..5の並べ替えが必要です".into());
    }
    Ok(())
}

fn plans(k: usize, incoming: i32, rules: &Rules) -> Vec<Plan> {
    let mut out = Vec::new();
    for &deadline in &rules.deadlines {
        for &target in &rules.main_targets {
            let plan = Plan { policy: Policy::Main, hands: deadline.min(k), target };
            if !out
                .iter()
                .any(|p: &Plan| p.hands == plan.hands && p.target == target && p.policy == plan.policy)
            {
                out.push(plan);
            }
        }
    }
    for count in [incoming, (incoming + 1) / 2] {
        let target = (count * OJAMA_RATE).max(1);
        if !out.iter().any(|p| p.policy == Policy::Counter && p.target == target) {
            out.push(Plan { policy: Policy::Counter, hands: k, target });
        }
    }
    out
}

fn elapsed(out: &Outcome, rules: &Rules) -> (u64, Option<u64>) {
    let mut time = 0;
    let mut fire = None;
    for step in &out.steps {
        time += rules.hand_ms;
        if step.score > 0 {
            fire = Some(time);
        }
        time += rules.chain_ms[step.chain as usize];
    }
    (time, fire)
}

/// 手の上限に加え、途中の小連鎖による停止時間も窓から差し引く。
fn timed_run(
    board: BitBoard,
    tape: &[Pair],
    mut plan: Plan,
    cfg: &Config,
    incoming: i32,
    window: u64,
    rules: &Rules,
    carry: i32,
    cache: &mut ama_f::Decisions,
) -> Outcome {
    loop {
        let out = ama_f::run_cached(board, tape, plan, cfg, incoming, true, carry, cache);
        let mut time = 0;
        let mut fits = 0;
        for step in &out.steps {
            time += rules.hand_ms;
            if time > window {
                break;
            }
            fits += 1;
            time += rules.chain_ms[step.chain as usize];
        }
        if fits == out.steps.len() {
            return out;
        }
        // Kが変わると選択も変わるため、切り貼りせず最初から再評価する。
        plan.hands = fits;
    }
}

fn after_fire(state: &State, side: usize, out: &Outcome, duration: u64) -> State {
    let mut next = state.clone();
    next.boards[side] = out.board;
    next.cursors[side] += out.steps.len();
    next.ready[side] += duration;
    next.pending[side] = out.pending;
    next.carry[side] = out.score_remainder;
    next.pending[1 - side] += out.excess;
    next
}

fn branch(
    state: &State,
    side: usize,
    level: usize,
    plan: Plan,
    window: u64,
    tape: &[Pair],
    cfg: &Config,
    rules: &Rules,
    cache: &mut ama_f::Decisions,
) -> Branch {
    let (mut node, next) = make_branch(state, side, level, plan, window, tape, cfg, rules, cache);
    descend(&mut node, next, tape, cfg, rules, cache);
    node
}

fn make_branch(
    state: &State, side: usize, level: usize, plan: Plan, window: u64,
    tape: &[Pair], cfg: &Config, rules: &Rules, cache: &mut ama_f::Decisions,
) -> (Branch, State) {
    let start = state.ready[side];
    let index = state.cursors[side];
    let (out, unavailable) = candidate_run(
        state.boards[side],
        &tape[index..],
        plan,
        cfg,
        state.pending[side],
        window,
        rules,
        state.carry[side],
        cache,
    );
    let (duration, fire) = elapsed(&out, rules);
    let next = after_fire(state, side, &out, duration);
    let node = Branch {
        completed_ms: None,
        progress_truncated: None,
        unavailable,
        preparation: None,
        side,
        level,
        plan,
        start_index: index,
        end_index: next.cursors[side],
        start_ms: start,
        fire_ms: fire.map(|t| start + t),
        end_ms: start + duration,
        available_hands: (window / rules.hand_ms).min(MAX_HANDS as u64) as usize,
        balance: None,
        landed: state.landed,
        pending: next.pending,
        truncated: out.interrupted,
        children: Vec::new(),
        outcome: out,
    };
    (node, next)
}

fn candidate_run(board: BitBoard, tape: &[Pair], plan: Plan, cfg: &Config,
    incoming: i32, window: u64, rules: &Rules, carry: i32, cache: &mut ama_f::Decisions,
) -> (Outcome, bool) {
    if plan.policy != Policy::QuickCounter {
        return (timed_run(board, tape, plan, cfg, incoming, window, rules, carry, cache), false);
    }
    let probe = ama_f::quick::probe(board, tape, plan.hands, incoming, plan.target / OJAMA_RATE);
    let unavailable = probe.best.is_none() || !probe.complete;
    let mut out = probe.best.unwrap_or_else(||
        ama_f::run(board, &[], Plan { hands: 0, ..plan }, cfg, incoming, true));
    if !probe.complete {
        out.interrupted = crate::ama_budget::expired();
        out.coverage = tape.len() >= plan.hands;
    }
    (out, unavailable)
}

fn descend(
    node: &mut Branch,
    mut state: State,
    tape: &[Pair],
    cfg: &Config,
    rules: &Rules,
    cache: &mut ama_f::Decisions,
) {
    let side = node.side;
    let other = 1 - side;
    // 半量対応の未相殺分はこの側の連鎖後に着地し、次の自分の盤面へ引き継ぐ。
    ama_f::settle(&mut node.outcome, &rules.garbage_order);
    state.boards[side] = node.outcome.board;
    state.landed[side] += node.outcome.fallen;
    state.pending[side] = node.outcome.pending;
    let start = node.fire_ms.unwrap_or(node.end_ms).max(state.ready[other]);
    let window = node.end_ms.saturating_sub(start);
    let k = (window / rules.hand_ms).min(MAX_HANDS as u64) as usize;
    if node.outcome.excess > 0
        && node.outcome.coverage
        && !node.outcome.dead
        && k > 0
        && node.level < rules.levels
    {
        state.ready[other] = start;
        node.children = plans(k, state.pending[other], rules)
            .into_iter()
            .map(|p| branch(&state, other, node.level + 1, p, window, tape, cfg, rules, cache))
            .collect();
    } else {
        node.truncated = node.outcome.excess > 0 && k > 0 && node.level == rules.levels;
        settle_leaf(node, &mut state, rules);
    }
    node.landed = state.landed;
    node.pending = state.pending;
}

fn settle_leaf(node: &mut Branch, state: &mut State, rules: &Rules) {
    let other = 1 - node.side;
    let mut receiver = ama_f::run(
        state.boards[other],
        &[],
        Plan { policy: Policy::Main, hands: 0, target: 0 },
        &Config {
            trigger: crate::ama_search::DEFAULT_TRIGGER,
            width: 1,
            depth: 2,
            colors: [1, 2, 3, 4],
            weight: crate::ama_eval::Weight::BUILD,
        },
        state.pending[other],
        true,
    );
    ama_f::settle(&mut receiver, &rules.garbage_order);
    state.landed[other] += receiver.fallen;
    state.pending[other] = receiver.pending;
    // 深さ打切りの未読反撃は truncated で区別。収支は残予告も含めた打切り時点の値。
    if node.outcome.coverage {
        node.balance = Some(state.landed[1] + state.pending[1] - state.landed[0] - state.pending[0]);
    }
}

pub fn evaluate(
    boards: [BitBoard; 2],
    tape: &[Pair],
    indices: [usize; 2],
    k: usize,
    incoming: i32,
    cfg: &Config,
    rules: &Rules,
) -> Vec<Branch> {
    let state = State {
        carry: [0, 0],
        boards,
        cursors: indices,
        ready: [0, 0],
        landed: [0, 0],
        pending: [incoming, 0],
    };
    let mut cache = ama_f::Decisions::new();
    plans(k, incoming, rules)
        .into_iter()
        .map(|p| branch(&state, 0, 1, p, k as u64 * rules.hand_ms, tape, cfg, rules, &mut cache))
        .collect()
}

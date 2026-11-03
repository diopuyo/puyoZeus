//! 既定OFFの幅優先評価。未読候補を残し、完了した候補だけを比較できるようにする。
use super::*;
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Options {
    pub opponent_busy_ms: u64,
    pub advance_opponent: bool,
    pub preparation_mode: String,
    pub opponent_horizon: Option<usize>,
    pub opponent_target: i32,
    pub quick_candidate: bool,
    pub quick_visible: usize,
    pub quick_min_return: i32,
    pub budget_ms: Option<u64>,
    // 試作の仮想時計。未指定なら従来どおり実時間を使う。
    pub budget_clock_scale: Option<f64>,
    // 同じ探索の途中状態を再現するレビュー専用の時計起点。
    pub progress_start: Option<Instant>,
    pub candidate_limit: Option<usize>,
    pub widths: Vec<usize>,
    pub threads: Option<usize>,
}

impl Options {
    pub fn validate(&self) -> Result<(), String> {
        if self.budget_clock_scale.is_some_and(|s| !s.is_finite() || s < 1.0) {
            return Err("予算時計の倍率は1以上の有限値です".into());
        }
        if self.opponent_busy_ms > MAX_TIME_MS || (self.opponent_busy_ms > 0 && !self.advance_opponent) {
            return Err("相手の初期連鎖待ちは0..120000ms、advance_opponentが必要です".into());
        }
        if self.quick_candidate && (!(2..=3).contains(&self.quick_visible)
            || !(1..=MAX_REQUEST_VALUE / OJAMA_RATE).contains(&self.quick_min_return)) {
            return Err("可視手2/3、返却閾値1..14285です".into());
        }
        if self.advance_opponent && (!matches!(self.preparation_mode.as_str(), "shared" | "separate" | "candidate")
            || self.opponent_horizon.is_some_and(|k| k == 0 || k > MAX_HANDS)
            || !(0..=MAX_REQUEST_VALUE).contains(&self.opponent_target)) {
            return Err("準備方式shared/separate/candidate、見通し1..128、目標0..1000000です".into());
        }
        if self.budget_ms.is_some_and(|n| n > MAX_TIME_MS)
            || self.candidate_limit.is_some_and(|n| !(2..=3).contains(&n))
            || self.widths.len() > 3
            || self.widths.iter().any(|&n| n == 0 || n > ama_f::MAX_WIDTH)
            || self.threads.is_some_and(|n| n != 2 && n != 4) {
            return Err("予算0..120000ms、本線数2/3、段別幅1..20000（最大3段）、スレッド数2/4です".into());
        }
        Ok(())
    }
    pub fn enabled(&self) -> bool {
        self.quick_candidate || self.advance_opponent || self.budget_ms.is_some() || self.candidate_limit.is_some() || !self.widths.is_empty()
    }
}

struct Job {
    state: State,
    side: usize,
    level: usize,
    plan: Plan,
    window: u64,
    parent: Option<usize>,
    promise: i32,
}

fn candidates(k: usize, incoming: i32, rules: &Rules, opt: &Options) -> Vec<Plan> {
    let all = plans(k, incoming, rules);
    let Some(limit) = opt.candidate_limit else { return all; };
    let main: Vec<_> = all.iter().filter(|p| p.policy == Policy::Main).copied().collect();
    let count = limit.min(main.len());
    let mut out = Vec::new();
    for i in 0..count {
        out.push(main[i * (main.len() - 1) / (count - 1).max(1)]);
    }
    out.extend(all.into_iter().filter(|p| p.policy == Policy::Counter).take(1));
    out
}

fn jobs(state: &State, side: usize, level: usize, k: usize, window: u64,
        parent: Option<usize>, promise: i32, rules: &Rules, opt: &Options) -> Vec<Job> {
    let mut plans = candidates(k, state.pending[side], rules, opt);
    if opt.quick_candidate && level == 1 {
        plans.push(Plan { policy: Policy::QuickCounter, hands: opt.quick_visible.min(k),
            target: opt.quick_min_return * OJAMA_RATE });
    }
    plans.into_iter().map(|plan| Job {
        state: state.clone(), side, level, plan, window, parent, promise,
    }).collect()
}

fn next_jobs(node: &mut Branch, mut state: State, rules: &Rules, opt: &Options,
             parent: usize) -> Vec<Job> {
    if node.unavailable || node.preparation.as_ref().is_some_and(|p| p.unsupported.is_some()) { return Vec::new(); }
    if node.outcome.interrupted {
        node.truncated = true;
        return Vec::new();
    }
    let side = node.side;
    let other = 1 - side;
    ama_f::settle(&mut node.outcome, &rules.garbage_order);
    state.boards[side] = node.outcome.board;
    state.landed[side] += node.outcome.fallen;
    state.pending[side] = node.outcome.pending;
    let start = node.fire_ms.unwrap_or(node.end_ms).max(state.ready[other]);
    let window = node.end_ms.saturating_sub(start);
    let k = (window / rules.hand_ms).min(MAX_HANDS as u64) as usize;
    let can_reply = node.outcome.excess > 0 && node.outcome.coverage && !node.outcome.dead && k > 0;
    let children = if can_reply && node.level < rules.levels {
        state.ready[other] = start;
        let promise = state.pending[other];
        jobs(&state, other, node.level + 1, k, window, Some(parent), promise, rules, opt)
    } else { Vec::new() };
    node.truncated = node.outcome.excess > 0 && k > 0 && node.level == rules.levels;
    if children.is_empty() {
        settle_leaf(node, &mut state, rules);
    } else if opt.budget_ms.is_some() {
        // 未読の子しか残らない場合に使う、既読部分だけの暫定収支。
        settle_leaf(node, &mut state.clone(), rules);
    }
    node.landed = state.landed;
    node.pending = state.pending;
    children
}

fn assemble(mut arena: Vec<(Branch, Option<usize>)>) -> Vec<Branch> {
    let mut roots = Vec::new();
    while let Some((mut node, parent)) = arena.pop() {
        node.children.reverse();
        if let Some(parent) = parent {
            if interrupted(&node) { arena[parent].0.truncated = true; }
            arena[parent].0.children.push(node);
        } else { roots.push(node); }
    }
    roots.reverse();
    roots
}

pub fn interrupted(node: &Branch) -> bool {
    node.outcome.interrupted || node.children.iter().any(interrupted)
}

pub fn evaluate(boards: [BitBoard; 2], tape: &[Pair], indices: [usize; 2], k: usize,
                incoming: i32, cfg: &Config, rules: &Rules, opt: &Options) -> Vec<Branch> {
    let state = State { carry: [0, 0], boards, cursors: indices, ready: [0, opt.opponent_busy_ms],
                        landed: [0, 0], pending: [incoming, 0] };
    let mut layer = jobs(&state, 0, 1, k, k as u64 * rules.hand_ms, None, 0, rules, opt);
    let mut arena = Vec::new();
    let mut cache = ama_f::Decisions::new();
    let mut shared = std::collections::HashMap::new();
    while !layer.is_empty() {
        // 浅い段を完了後、前段の余剰量・必要量を満たす速さで優先する。
        if opt.budget_ms.is_some() {
            layer.sort_by_key(|j| (std::cmp::Reverse(j.promise),
                j.plan.policy != Policy::Counter,
                std::cmp::Reverse(j.plan.target / j.plan.hands.max(1) as i32)));
        }
        let mut next = Vec::new();
        for job in layer {
            let width = opt.widths.get(job.level - 1).or(opt.widths.last()).copied().unwrap_or(cfg.width);
            let cfg = Config { width, ..*cfg };
            let (mut node, mut state) = make_branch(&job.state, job.side, job.level, job.plan,
                job.window, tape, &cfg, rules, &mut cache);
            if opt.advance_opponent && job.level == 1 && !node.unavailable {
                prepare::apply(&mut node, &mut state, &job.state, tape, &cfg, rules, opt, k, &mut shared);
            }
            next.extend(next_jobs(&mut node, state, rules, opt, arena.len()));
            if let Some(start) = opt.progress_start {
                node.completed_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                node.progress_truncated = Some(node.truncated);
            }
            arena.push((node, job.parent));
        }
        layer = next;
    }
    assemble(arena)
}

pub fn directions(boards: [BitBoard; 2], tape: &[Pair], indices: [usize; 2], k: usize,
    incoming: i32, cfg: &Config, rules: &Rules, opt: &Options, both: bool, parallel: bool,
) -> Result<(Vec<Branch>, Vec<Branch>), String> {
    let deadline = opt.budget_ms.map(|ms| Instant::now() + match opt.budget_clock_scale {
        Some(scale) => Duration::from_secs_f64(ms as f64 / 1000.0 / scale),
        None => Duration::from_millis(ms),
    });
    let calculate = |flip: bool| crate::ama_budget::within(deadline, || {
        let (b, i) = if flip { ([boards[1], boards[0]], [indices[1], indices[0]]) } else { (boards, indices) };
        if opt.enabled() { evaluate(b, tape, i, k, incoming, cfg, rules, opt) }
        else { super::evaluate(b, tape, i, k, incoming, cfg, rules) }
    });
    let run = || if both && parallel {
        rayon::join(|| calculate(false), || calculate(true))
    } else { (calculate(false), if both { calculate(true) } else { Vec::new() }) };
    if let Some(threads) = opt.threads {
        // 既存の本番プールのスレッド数は変更しない。
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| e.to_string())?;
        Ok(pool.install(run))
    } else { Ok(run()) }
}

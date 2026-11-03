//! 方式F。本物テープは遷移に使い、手の選択へ渡すのは常に最大3組。
//! 既存の本番探索への配線は行わない。収支は個数の生値で返す。
use crate::ama_search::{self, Config, Pair, Placement, VISIBLE_PAIRS};
use crate::bitboard::{
    drop_one, height_of, BitBoard, BOARD_COLS, COLOR_OJAMA, COLOR_PURPLE, COLOR_RED, DEATH_BIT_INDEX,
    DEATH_COL,
};
use std::collections::HashMap;

pub const OJAMA_RATE: i32 = 70;
pub const MAX_DROP: i32 = 30;
pub const MAX_HANDS: usize = 128;
pub const MAX_WIDTH: usize = 20000;
pub const MAX_REQUEST_VALUE: i32 = 1000000;
pub type Decisions = HashMap<Vec<u16>, Option<Placement>>;
#[path = "ama_quick.rs"]
pub mod quick;
#[path = "ama_growth.rs"]
pub mod growth;
#[path = "ama_race.rs"]
pub mod race;
#[path = "ama_race_detail.rs"]
pub mod race_detail;
#[path = "ama_branches.rs"]
pub mod branches;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Main,
    Counter,
    QuickCounter,
}

#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub policy: Policy,
    pub hands: usize,
    pub target: i32,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub hand: usize,
    pub placement: Placement,
    pub score: i32,
    pub chain: i32,
    pub board: BitBoard,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub carry_in: i32,
    pub score_remainder: i32,
    pub fired: bool,
    pub reached: bool,
    pub fire_hand: Option<usize>,
    pub score: i32,
    pub max_score: i32,
    pub total_score: i32,
    pub cancelled: i32,
    pub excess: i32,
    pub fallen: i32,
    pub pending: i32,
    pub overflow: i32,
    pub board: BitBoard,
    pub steps: Vec<Step>,
    pub dead: bool,
    pub coverage: bool,
    pub interrupted: bool,
}

pub fn validate(board: &BitBoard, tape: &[Pair], cfg: &Config) -> Result<(), String> {
    if cfg.width == 0 || cfg.width > MAX_WIDTH || cfg.depth < 2 || cfg.depth > MAX_HANDS {
        return Err("幅は1..20000、深さは2..128が必要です".into());
    }
    let mut colors = cfg.colors;
    colors.sort();
    if colors.iter().any(|c| !(COLOR_RED..=COLOR_PURPLE).contains(c))
        || colors.windows(2).any(|p| p[0] == p[1])
    {
        return Err("試合色は異なる4色（1..5）で指定してください".into());
    }
    if tape.iter().any(|p| !colors.contains(&p.0) || !colors.contains(&p.1)) {
        return Err("本物ツモに試合色以外の色があります".into());
    }
    for (i, p) in board.colors.iter().enumerate() {
        if !colors.contains(&(i as u8 + 1)) && p.iter().any(|&v| v != 0) {
            return Err("盤面に試合色以外の色があります".into());
        }
    }
    Ok(())
}

fn initial(board: BitBoard, incoming: i32) -> Outcome {
    Outcome {
        carry_in: 0,
        score_remainder: 0,
        fired: false,
        reached: false,
        fire_hand: None,
        score: 0,
        max_score: 0,
        total_score: 0,
        cancelled: 0,
        excess: 0,
        fallen: 0,
        pending: incoming,
        overflow: 0,
        board,
        steps: Vec::new(),
        dead: height_of(&board, DEATH_COL) > DEATH_BIT_INDEX,
        coverage: true,
        interrupted: false,
    }
}

fn record(out: &mut Outcome, p: Placement, pair: Pair, incoming: i32) {
    let result = ama_search::apply(&out.board, p, pair).expect("選択済みの生存手");
    out.board = result.final_board;
    let score = result.exact_score as i32;
    out.total_score += score;
    out.max_score = out.max_score.max(score);
    if score > 0 {
        out.score = score;
        out.fired = true;
        out.fire_hand = Some(out.steps.len() + 1);
    }
    let sent = (out.total_score + out.carry_in) / OJAMA_RATE;
    out.score_remainder = (out.total_score + out.carry_in) % OJAMA_RATE;
    out.cancelled = sent.min(incoming);
    out.excess = (sent - incoming).max(0);
    out.pending = (incoming - sent).max(0);
    out.steps.push(Step {
        hand: out.steps.len() + 1,
        placement: p,
        score,
        chain: result.chain_count,
        board: out.board,
    });
}

/// おじゃまの端数位置は入力順で固定。1回の落下は最大30個、残りは予告に残す。
pub fn settle(out: &mut Outcome, order: &[usize; BOARD_COLS]) {
    let count = out.pending.min(MAX_DROP);
    for i in 0..count as usize {
        if let Some(b) = drop_one(&out.board, order[i % BOARD_COLS], COLOR_OJAMA) {
            out.board = b;
        } else {
            out.overflow += 1;
        }
    }
    out.fallen += count;
    out.pending -= count;
    out.dead |= out.overflow > 0 || height_of(&out.board, DEATH_COL) > DEATH_BIT_INDEX;
}

/// stop=false はオラクル互換のK手継続。stop=true は最初の実発火で終了。
/// 目標未達の小連鎖も実際には発火するため、無視して次の大連鎖へ飛ばさない。
pub fn run(board: BitBoard, tape: &[Pair], plan: Plan, cfg: &Config, incoming: i32, stop: bool) -> Outcome {
    #[cfg(feature = "ama-profile")]
    let _timer = crate::ama_profile_clock::Timer::new(9);
    run_cached(board, tape, plan, cfg, incoming, stop, 0, &mut Decisions::new())
}

fn decision(
    board: &BitBoard,
    visible: &[Pair],
    remaining: usize,
    cfg: &Config,
    need: i32,
    cache: &mut Decisions,
) -> Option<Placement> {
    let mut key: Vec<u16> = board.colors.iter().flatten().copied().chain(board.ojama).collect();
    key.extend([
        cfg.width as u16,
        cfg.depth as u16,
        remaining as u16,
        need as u16,
        (need as u32 >> 16) as u16,
        cfg.trigger as u16,
        (cfg.trigger as u32 >> 16) as u16,
    ]);
    key.extend(visible.iter().map(|&(a, b)| a as u16 | ((b as u16) << 8)));
    if let Some(&found) = cache.get(&key) { return found; }
    let found = ama_search::decide(board, visible, remaining, cfg, need);
    if !crate::ama_budget::expired() { cache.insert(key, found); }
    found
}

/// cfg固定の1回の候補木内だけで使う。同じ盤面・可視ツモ・残手数の決定を共有する。
pub fn run_cached(
    board: BitBoard,
    tape: &[Pair],
    plan: Plan,
    cfg: &Config,
    incoming: i32,
    stop: bool,
    carry: i32,
    cache: &mut Decisions,
) -> Outcome {
    let mut out = initial(board, incoming);
    out.carry_in = carry;
    out.score_remainder = carry;
    if out.dead {
        return out;
    }
    // 本線の大きさは原典の探索打切り得点を変えて候補化する。0はオラクル設定。
    let cfg = Config {
        trigger: if plan.policy == Policy::Main && plan.target > 0 { plan.target } else { cfg.trigger },
        ..*cfg
    };
    for i in 0..plan.hands {
        if crate::ama_budget::expired() { out.interrupted = true; break; }
        let visible_count = VISIBLE_PAIRS.min(plan.hands - i);
        if i + visible_count > tape.len() {
            out.coverage = false;
            break;
        }
        let visible = &tape[i..i + visible_count];
        let need = if plan.policy == Policy::Counter { (plan.target - carry).max(1) } else { 0 };
        let chosen = decision(&out.board, visible, plan.hands - i, &cfg, need, cache);
        if crate::ama_budget::expired() { out.interrupted = true; break; }
        let Some(p) = chosen else {
            out.dead = true;
            break;
        };
        record(&mut out, p, tape[i], incoming);
        let score = out.steps.last().unwrap().score;
        if score > 0 {
            let credit = if plan.policy == Policy::Counter { carry } else { 0 };
            out.reached |= score + credit >= plan.target;
            if stop {
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Config {
        Config {
            trigger: ama_search::DEFAULT_TRIGGER,
            width: 4,
            depth: 6,
            colors: [1, 2, 3, 4],
            weight: crate::ama_eval::Weight::BUILD,
        }
    }
    #[test]
    fn future_is_not_visible_to_first_decision() {
        let cfg = config();
        let b = BitBoard::default();
        let a = run(b, &[(1, 2); 8], Plan { policy: Policy::Main, hands: 8, target: 0 }, &cfg, 0, false);
        let mut tape = [(1, 2); 8];
        tape[3..].fill((3, 4));
        let c = run(b, &tape, Plan { policy: Policy::Main, hands: 8, target: 0 }, &cfg, 0, false);
        assert_eq!(a.steps[0].placement, c.steps[0].placement);
    }
    #[test]
    fn shortage_is_explicit() {
        let out = run(
            BitBoard::default(),
            &[(1, 2)],
            Plan { policy: Policy::Main, hands: 4, target: 0 },
            &config(),
            0,
            true,
        );
        assert!(!out.coverage);
        assert!(!out.fired);
        assert!(out.steps.is_empty());
    }
    #[test]
    fn garbage_cap_and_remainder() {
        let mut out = initial(BitBoard::default(), 37);
        settle(&mut out, &[0, 1, 2, 3, 4, 5]);
        assert_eq!((out.fallen, out.pending, out.overflow), (30, 7, 0));
        assert_eq!(height_of(&out.board, 2), 5);
    }

    #[test]
    fn score_remainder_is_carried_into_next_counter() {
        let mut b = BitBoard::default();
        for x in 0..3 {
            b.colors[0][x] = 1;
        }
        let out = run_cached(
            b,
            &[(1, 2)],
            Plan { policy: Policy::Counter, hands: 1, target: 70 },
            &config(),
            1,
            true,
            30,
            &mut Decisions::new(),
        );
        assert_eq!((out.score, out.cancelled, out.score_remainder), (40, 1, 0));
        assert!(out.reached);
    }
}

//! 催促時の可視手全列挙。最初の発火を飛ばさず、個数の生値を保持する。
use super::*;

pub const CONSUMPTION_LIMIT: f64 = 0.6;

#[derive(Clone, Debug)]
pub struct Probe {
    pub best: Option<Outcome>,
    pub maximum_return: i32,
    pub expanded: usize,
    pub complete: bool,
}

/// 発火した経路はその場で止める。小連鎖後の次の組を即座に置く近道は作らない。
fn enumerate(out: &Outcome, tape: &[Pair], limit: usize, incoming: i32,
             threshold: i32, probe: &mut Probe) {
    if out.steps.len() == limit || out.dead { return; }
    if crate::ama_budget::expired() { probe.complete = false; return; }
    let pair = tape[out.steps.len()];
    for p in ama_search::placements(&out.board, pair) {
        if crate::ama_budget::expired() { probe.complete = false; return; }
        if ama_search::apply(&out.board, p, pair).is_none() { continue; }
        probe.expanded += 1;
        let mut next = out.clone();
        record(&mut next, p, pair, incoming);
        if !next.fired {
            enumerate(&next, tape, limit, incoming, threshold, probe);
            continue;
        }
        let returned = next.cancelled + next.excess;
        probe.maximum_return = probe.maximum_return.max(returned);
        if returned < threshold { continue; }
        // 最短、同手数では得点の小さい発火、同点では列挙順。
        let rank = (next.steps.len(), next.total_score);
        if probe.best.as_ref().is_none_or(|b| rank < (b.steps.len(), b.total_score)) {
            next.reached = true;
            probe.best = Some(next);
        }
    }
}

pub fn probe(board: BitBoard, tape: &[Pair], visible: usize, incoming: i32,
             threshold: i32) -> Probe {
    let mut result = Probe { best: None, maximum_return: 0, expanded: 0,
                             complete: tape.len() >= visible };
    enumerate(&initial(board, incoming), tape, visible.min(tape.len()), incoming,
              threshold, &mut result);
    result
}

pub fn run(board: BitBoard, tape: &[Pair], plan: Plan, cfg: &Config, incoming: i32,
           consumption: f64, visible: usize, threshold: i32) -> (Outcome, Probe, bool) {
    let inspect = consumption < CONSUMPTION_LIMIT;
    let result = if inspect { probe(board, tape, visible.min(plan.hands), incoming, threshold) }
        else { Probe { best: None, maximum_return: 0, expanded: 0, complete: true } };
    if result.complete {
        if let Some(out) = result.best.clone() { return (out, result, true); }
    }
    let out = super::run(board, tape, plan, cfg, incoming, true);
    (out, result, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_board_cannot_return_nine_in_two_pairs() {
        let p = probe(BitBoard::default(), &[(1, 2); 2], 2, 60, 9);
        assert!(p.complete && p.best.is_none());
        assert!(p.expanded > 22);
    }
    #[test]
    fn missing_visible_pair_is_not_a_negative_proof() {
        assert!(!probe(BitBoard::default(), &[(1, 2)], 2, 0, 9).complete);
    }
    #[test]
    fn exhausted_budget_is_not_a_negative_proof() {
        let p = crate::ama_budget::within(Some(std::time::Instant::now()), ||
            probe(BitBoard::default(), &[(1, 2); 2], 2, 0, 9));
        assert!(!p.complete);
    }
}

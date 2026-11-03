//! 配信用の6枝探索。最大連鎖得点と初手別の代表道筋を保持する。
use crate::{chain, eval, field::{Field, Placement}, order, table::Table};

const BAGS: [[u8; 4]; 6] = [[4, 6, 7, 5], [4, 7, 6, 5], [4, 5, 6, 7],
    [6, 7, 4, 5], [6, 5, 4, 7], [7, 5, 4, 6]];
const PRUNE_SCORE: i32 = 5000;

#[derive(Clone, Copy)]
pub struct Config { pub width: usize, pub depth: usize, pub trigger: i32 }

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub placement: Placement, pub score: i64, pub root_score: i32,
    pub search_score: i32, pub path: Vec<Placement>,
}

/// layer_max[k] = k+1 手目を置いた時に撃てた連鎖の得点の最大 (その層の全ノード、刈る前の値)。
#[derive(Default, Debug, PartialEq)]
pub struct Result { pub candidates: Vec<Candidate>, pub expanded: usize, pub layer_max: Vec<i64>, arena: Vec<Step> }

#[derive(Debug, PartialEq)]
struct Step { parent: Option<usize>, placement: Placement }

#[derive(Clone)]
struct Node { field: Field, action: i32, eval: i32, root: usize, path: usize }

/// 満杯の層で「合計が最小の最初の位置」(同点は前の位置) を保つトーナメント木。
/// 置き換えのたびに全部を見直す代わりに、置き換えた葉から根まで (8 段) だけ直す。
struct Layer { data: Vec<Node>, table: Table, width: usize, tree: Vec<usize>, leaves: usize, built: bool }

const NO_NODE: usize = usize::MAX;

/// a と b のうち弱い方。同点は a (木の左 = 前の位置) を返す。
fn weaker(data: &[Node], a: usize, b: usize) -> usize {
    if a == NO_NODE { return b; }
    if b == NO_NODE { return a; }
    if data[b].total() < data[a].total() { b } else { a }
}

impl Layer {
    fn new(width: usize) -> Self {
        let width = width.max(1);
        let leaves = width.next_power_of_two();
        Self { data: Vec::new(), table: Table::default(), width, tree: vec![NO_NODE; 2 * leaves], leaves, built: false }
    }
    fn build(&mut self) {
        self.tree.fill(NO_NODE);
        for i in 0..self.data.len() { self.tree[self.leaves + i] = i; }
        for k in (1..self.leaves).rev() { self.tree[k] = weaker(&self.data, self.tree[2 * k], self.tree[2 * k + 1]); }
        self.built = true;
    }
    fn update(&mut self, i: usize) {
        let mut k = (self.leaves + i) / 2;
        while k >= 1 {
            self.tree[k] = weaker(&self.data, self.tree[2 * k], self.tree[2 * k + 1]);
            k /= 2;
        }
    }
    fn add(&mut self, node: Node) {
        if self.data.len() < self.width { self.data.push(node); self.built = false; return; }
        if !self.built { self.build(); }
        let weakest = self.tree[1];
        if self.data[weakest].total() < node.total() { self.data[weakest] = node; self.update(weakest); }
    }
    fn clear(&mut self) { self.data.clear(); self.built = false; self.table.update(); }
}

impl Node { fn total(&self) -> i32 { self.action + self.eval } }

pub fn queue(visible: &[[u8; 2]], branch: usize, depth: usize) -> Vec<[u8; 2]> {
    let mut q = visible.to_vec();
    let bag = BAGS[branch % BAGS.len()];
    for i in 0..depth.saturating_sub(q.len()) {
        let offset = (i % 2) * 2;
        q.push([bag[offset], bag[offset + 1]]);
    }
    q
}

/// 置いた後の盤 (placed) の連鎖を進めて子を作る。
fn apply(parent: &Node, p: Placement, mut field: Field) -> Option<(Node, i32)> {
    let (chains, score) = chain::simulate_stats(&mut field);
    if field.dead() { return None; }
    let tear = parent.field.heights[p.x] != parent.field.heights[p.child()?];
    let mut child = parent.clone();
    child.field = field;
    child.action += eval::action(tear, chains);
    Some((child, score))
}

fn roots(field: Field, pair: [u8; 2], layer: &mut Layer, result: &mut Result) {
    let root = Node { field, action: 0, eval: 0, root: 0, path: 0 };
    let mut best = 0i64;
    field.each_child(pair, |p, next| {
        result.expanded += 1;
        let Some((mut child, score)) = apply(&root, p, next) else { return; };
        best = best.max(score as i64);
        child.root = result.candidates.len();
        child.path = result.arena.len();
        result.arena.push(Step { parent: None, placement: p });
        child.eval = eval::evaluate(&child.field);
        result.candidates.push(Candidate { placement: p, score: score as i64,
            root_score: score, search_score: child.total(), path: vec![p] });
        layer.data.push(child);
    });
    result.layer_max.push(best);
}

fn path_from(arena: &[Step], parent: usize, placement: Placement) -> Vec<Placement> {
    let mut path = vec![placement];
    let mut id = Some(parent);
    while let Some(i) = id {
        path.push(arena[i].placement);
        id = arena[i].parent;
    }
    path.reverse();
    path
}

fn accept(mut child: Node, p: Placement, score: i32, children: &mut Layer, result: &mut Result) {
    let candidate = &mut result.candidates[child.root];
    if score as i64 > candidate.score { candidate.path = path_from(&result.arena, child.path, p); }
    candidate.score = candidate.score.max(score as i64);
    if score >= PRUNE_SCORE { return; }
    let hash = child.field.hash();
    let (bucket, slot, entry) = children.table.get(hash);
    child.eval = if let Some(e) = entry {
        if child.action < e.action as i32 { return; }
        e.eval
    } else { eval::evaluate(&child.field) };
    candidate.search_score = candidate.search_score.max(child.total());
    children.table.set(bucket, slot, hash, child.action, child.eval);
    let step = Step { parent: Some(child.path), placement: p };
    child.path = result.arena.len();
    result.arena.push(step);
    children.add(child);
}

fn think(pair: [u8; 2], parents: &mut Layer, children: &mut Layer, result: &mut Result) {
    order::sort(&mut parents.data, |a, b| a.total() > b.total());
    let mut best = 0i64;
    for parent in &parents.data {
        parent.field.each_child(pair, |p, next| {
            result.expanded += 1;
            if let Some((child, score)) = apply(parent, p, next) {
                best = best.max(score as i64);
                accept(child, p, score, children, result);
            }
        });
    }
    result.layer_max.push(best);
    parents.clear();
}

pub fn search(field: Field, queue: &[[u8; 2]], config: Config) -> Result {
    let mut result = Result::default();
    let depth = config.depth.min(queue.len());
    if queue.len() < 2 || depth == 0 { return result; }
    let (mut a, mut b) = (Layer::new(config.width), Layer::new(config.width));
    roots(field, queue[0], &mut a, &mut result);
    if result.candidates.is_empty() { return result; }
    for &pair in &queue[1..depth] {
        think(pair, &mut a, &mut b, &mut result);
        std::mem::swap(&mut a, &mut b);
        if result.candidates.iter().any(|c| c.score >= config.trigger as i64) { break; }
    }
    result.arena = Vec::new();
    result
}

/// 枝ごとの探索は独立。完了順にかかわらず枝番号順で返す。
pub fn branches(field: Field, visible: &[[u8; 2]], config: Config,
    count: usize, threads: usize) -> Vec<Result> {
    if count == 0 { return Vec::new(); }
    let workers = threads.max(1).min(count);
    if workers == 1 {
        return (0..count).map(|b| search(field, &queue(visible, b, config.depth), config)).collect();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers).map(|worker| scope.spawn(move || {
            (worker..count).step_by(workers).map(|b| {
                (b, search(field, &queue(visible, b, config.depth), config))
            }).collect::<Vec<_>>()
        })).collect();
        let mut slots: Vec<Option<Result>> = (0..count).map(|_| None).collect();
        for handle in handles {
            for (b, result) in handle.join().expect("ama探索スレッド異常終了") { slots[b] = Some(result); }
        }
        slots.into_iter().map(|r| r.unwrap()).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn six_branches_and_visible_prefix() {
        let visible = [[5, 4], [6, 7], [4, 4]];
        for b in 0..6 {
            let q = queue(&visible, b, 7);
            assert_eq!(&q[..3], &visible);
            assert_eq!(q[3], [BAGS[b][0], BAGS[b][1]]);
            assert_eq!(q[5], q[3]);
        }
    }
    #[test]
    fn weakest_tree_matches_full_scan() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for width in [1usize, 2, 3, 7, 8, 250] {
            let mut layer = Layer::new(width);
            let mut naive: Vec<i32> = Vec::new();
            for _ in 0..3000 {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                let total = (state % 50) as i32; // 同点を多くする
                layer.add(Node { field: Field::default(), action: total, eval: 0, root: 0, path: 0 });
                if naive.len() < width { naive.push(total); } else {
                    let mut first = 0;
                    for i in 1..naive.len() { if naive[i] < naive[first] { first = i; } }
                    if naive[first] < total { naive[first] = total; }
                }
                assert_eq!(layer.data.iter().map(Node::total).collect::<Vec<_>>(), naive);
            }
        }
    }

    #[test]
    fn depth_one_keeps_all_roots() {
        let r = search(Field::default(), &[[4, 5], [4, 5]], Config { width: 1, depth: 1, trigger: 95000 });
        assert_eq!(r.candidates.len(), 22);
        assert_eq!(r.expanded, 22);
        assert!(r.candidates.iter().all(|c| c.path == vec![c.placement]));
    }

    #[test]
    fn parallel_branches_keep_exact_results() {
        let config = Config { width: 4, depth: 3, trigger: 95000 };
        let visible = [[4, 5], [6, 7]];
        let serial = branches(Field::default(), &visible, config, 6, 1);
        assert_eq!(serial, branches(Field::default(), &visible, config, 6, 2));
    }
}

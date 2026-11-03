//! Pythonを通さず保存入力を測る独立ベンチ。本番へ接続しない。
#![allow(dead_code)]
mod ama_eval;
mod ama_budget;
mod ama_form;
mod ama_order;
mod ama_search;
mod ama_table;
mod ama_weights;
mod bitboard;
mod ama_profile_clock;
use std::io::{self, BufRead};
use std::time::Instant;
use ama_search::{Config, Pair};

#[cfg(feature = "ama-profile-detail")]
#[path = "ama_profile_alloc.rs"]
mod ama_profile_alloc;

const BAGS: [[usize; 4]; 6] = [
    [0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2],
    [1, 2, 0, 3], [1, 3, 0, 2], [2, 3, 0, 1],
];

#[derive(serde::Deserialize)]
struct Input {
    board: Vec<u8>,
    known: Vec<Pair>,
    colors: [u8; 4],
    #[serde(default)]
    seed: usize,
    #[serde(default)]
    tape: Vec<Pair>,
}

fn branches(input: &Input, cfg: &Config, parallel: bool) -> Vec<Vec<ama_search::Candidate>> {
    use rayon::prelude::*;
    let board = bitboard::board_from_grid(&input.board);
    let run = |bag: &[usize; 4]| {
        let mut queue = input.known.clone();
        let start = queue.len();
        while queue.len() < cfg.depth {
            let offset = (queue.len() - start) % 2 * 2;
            queue.push((cfg.colors[bag[offset]], cfg.colors[bag[offset + 1]]));
        }
        ama_profile_clock::take();
        #[cfg(feature = "ama-profile-detail")]
        ama_profile_alloc::take();
        let result = ama_search::search(board, &queue, cfg);
        #[cfg(feature = "ama-profile-detail")]
        {
            let allocations = ama_profile_alloc::take();
            let stages = ama_profile_clock::take();
            eprintln!("{}", serde_json::json!({"thread":format!("{:?}", std::thread::current().id()),
                "stages":stages, "allocations":allocations}));
        }
        result
    };
    if parallel { BAGS.par_iter().map(run).collect() }
    else { BAGS.iter().map(run).collect() }
}

fn play(mut input: Input, cfg: &Config) {
    const TURNS: usize = 50;
    const VISIBLE: usize = 2;
    let mut board = bitboard::BitBoard::default();
    for turn in 0..TURNS {
        input.board = bitboard::board_to_grid(&board);
        input.known = input.tape[turn..turn+VISIBLE].to_vec();
        let began = Instant::now();
        let found = branches(&input, cfg, true);
        let mut total = found[0].clone();
        for branch in &found[1..] {
            for (a, b) in total.iter_mut().zip(branch) { a.score += b.score; }
        }
        ama_order::insertion(&mut total, |c| c.score);
        let elapsed = began.elapsed().as_secs_f64()*1000.;
        let Some(chosen) = total.first() else { break; };
        let Some(result) = ama_search::apply(&board, chosen.placement, input.tape[turn]) else { break; };
        println!("{}", serde_json::json!({"seed":input.seed,"turn":turn+1,"ms":elapsed,
            "score":result.exact_score,"chain":result.chain_count,"placement":chosen.placement,
            "candidates":total.iter().map(|c| (c.placement, c.score)).collect::<Vec<_>>(),
            "board_before":input.board,"known":input.known,"colors":input.colors,
            "final_board":bitboard::board_to_grid(&result.final_board)}));
        board = result.final_board;
    }
}

fn measure_six(input: &Input, index: usize, weight: ama_eval::Weight) {
    let cfg = Config { trigger: 80000, width: 250, depth: 16, colors: input.colors, weight };
    let began = Instant::now();
    let result = branches(input, &cfg, true);
    println!("{}", serde_json::json!({"input":index,"width":cfg.width,"depth":cfg.depth,
        "queues":6,"parallel":true,"ms":began.elapsed().as_secs_f64()*1000.,
        "candidates":result.iter().map(|v| v.iter().map(|c| (c.placement, c.score))
            .collect::<Vec<_>>()).collect::<Vec<_>>()}));
}

fn main() {
    // 初期化・I/Oは思考時間の外。6本版は参照との差を測る専用経路。
    rayon::ThreadPoolBuilder::new().num_threads(6).build_global().unwrap();
    let weight = ama_weights::load(None, "build").unwrap().weight;
    for (index, line) in io::stdin().lock().lines().enumerate() {
        let input: Input = serde_json::from_str(&line.unwrap()).unwrap();
        if std::env::args().any(|a| a == "--six") {
            measure_six(&input, index, weight);
            continue;
        }
        if !input.tape.is_empty() {
            let cfg = Config { trigger: 80000, width: 250, depth: 16, colors: input.colors, weight };
            play(input, &cfg);
            continue;
        }
        let widths = if std::env::args().any(|a| a == "--shallow") { vec![1] } else { vec![1, 4, 32, 250] };
        for width in widths {
            let depth = if width == 1 { 2 } else { 16 };
            let cfg = Config { trigger: 80000, width, depth, colors: input.colors, weight };
            let board = bitboard::board_from_grid(&input.board);
            ama_profile_clock::take();
            let began = Instant::now();
            let result = ama_search::multi(board, &input.known, depth, &cfg, 0);
            println!("{}", serde_json::json!({"input":index,"width":width,"depth":depth,
                "queues":2,"parallel":false,"ms":began.elapsed().as_secs_f64()*1000.,
                "stages_eval_quiet_table_apply":ama_profile_clock::take(),
                "scores":result.iter().map(|c| c.score).collect::<Vec<_>>()}));
            if width != 250 { continue; }
            for parallel in [false, true] {
                let began = Instant::now();
                let result = branches(&input, &cfg, parallel);
                println!("{}", serde_json::json!({"input":index,"width":width,"depth":depth,
                    "queues":6,"parallel":parallel,"ms":began.elapsed().as_secs_f64()*1000.,
                    "scores":result.iter().map(|v| v.iter().map(|c| c.score).collect::<Vec<_>>()).collect::<Vec<_>>()}));
            }
        }
    }
}

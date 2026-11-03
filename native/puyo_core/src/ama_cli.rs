//! 一致検証用の独立ランナー。配信や既存APIからは呼ばない。
#![allow(dead_code)]
mod ama_eval;
mod ama_budget;
mod ama_form;
mod ama_order;
mod ama_search;
mod ama_table;
mod ama_weights;
mod bitboard;
#[cfg(feature = "ama-profile")]
mod ama_profile_clock;
use ama_search::{Config, Pair};
use std::io::{self, BufRead, Write};

fn parse(text: &str) -> Vec<u8> {
    text.bytes().map(|b| b - b'0').collect()
}

fn run(args: &[&str], cfg: &Config) {
    let k = args[1].parse::<usize>().unwrap();
    let mut board = bitboard::board_from_grid(&parse(args[2]));
    let colors: [u8; 4] = parse(args[3]).try_into().unwrap();
    let pairs: Vec<Pair> = parse(args[4]).chunks_exact(2).map(|p| (p[0], p[1])).collect();
    let config = Config { colors, ..*cfg };
    let start = std::time::Instant::now();
    if args[0] == "SEARCH" {
        for c in ama_search::multi(board, &pairs, k, &config, 0) {
            println!("C {} {} {}", c.placement.0, c.placement.1, c.score);
        }
    } else if args[0] == "EVAL" {
        println!(
            "V {} {}",
            ama_eval::evaluate(&board, &colors, &cfg.weight),
            ama_table::hash(&board, &colors)
        );
    } else {
        let mut best = 0;
        for i in 0..k {
            let Some(p) = ama_search::decide(&board, &pairs[i..(i + 3).min(pairs.len())], k - i, &config, 0)
            else {
                break;
            };
            let r = ama_search::apply(&board, p, pairs[i]).unwrap();
            board = r.final_board;
            best = best.max(r.exact_score);
            println!("T {} {} {} {}", i + 1, p.0, p.1, r.exact_score);
        }
        println!("R {} {}", best, start.elapsed().as_nanos());
    }
    println!("END");
}

fn main() {
    let mut cfg = Config {
        trigger: ama_search::DEFAULT_TRIGGER,
        width: 8,
        depth: 6,
        colors: [1, 2, 3, 4],
        weight: ama_weights::load(None, "build").expect("同梱重み").weight,
    };
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let args: Vec<_> = line.split_whitespace().collect();
        match args.first().copied() {
            Some("CFG") => {
                cfg.width = args[1].parse().unwrap();
                cfg.depth = args[2].parse().unwrap();
                let profile = if args.get(4) == Some(&"ac") { "ac" } else { "build" };
                cfg.weight = ama_weights::load(None, profile).expect("同梱重み").weight;
                if args[3] == "0" {
                    cfg.weight.form = 0;
                }
                println!("OK");
            }
            Some("QUIT") => break,
            Some(_) => run(&args, &cfg),
            None => (),
        }
        io::stdout().flush().unwrap();
    }
}

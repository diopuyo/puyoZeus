use ama_reference::field::{Field, JSON_COLORS};
use serde_json::{json, Value};
use std::io::{self, BufRead};

fn describe(f: &Field) -> Value {
    let mut after = *f;
    let scores = ama_reference::chain::simulate(&mut after);
    json!({"hash": f.hash(), "board": f.grid(), "heights": f.heights, "dead": f.dead(),
        "trace": {"scores": scores, "board": after.grid(), "dead": after.dead()},
        "links": ama_reference::chain::links(f),
        "eval": ama_reference::eval::features(f)})
}

fn main() {
    for line in io::stdin().lock().lines() {
        let input: Value = serde_json::from_str(&line.unwrap()).unwrap();
        if let Some(array) = input["sort"].as_array() {
            let mut values: Vec<(i64, usize)> = array.iter().enumerate()
                .map(|(i, v)| (v.as_i64().unwrap(), i)).collect();
            ama_reference::order::sort(&mut values, |a, b| a.0 > b.0);
            println!("{}", json!(values.iter().map(|v| v.1).collect::<Vec<_>>()));
            continue;
        }
        let board: Vec<u8> = serde_json::from_value(input["board"].clone()).unwrap();
        let field = Field::from_json(&board).unwrap();
        let pair = [JSON_COLORS[input["visible"][0][0].as_u64().unwrap() as usize],
                    JSON_COLORS[input["visible"][0][1].as_u64().unwrap() as usize]];
        let mut out = describe(&field);
        out["moves"] = Value::Array(field.placements(pair).iter().map(|&p| {
            let mut item = describe(&field.place(p, pair).unwrap());
            item["x"] = json!(p.x); item["r"] = json!(p.r);
            item
        }).collect());
        println!("{out}");
    }
}

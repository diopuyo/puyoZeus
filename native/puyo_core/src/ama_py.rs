//! 方式F専用の明示呼出API。既存Pythonブリッジや採用フラグへは接続しない。
use crate::{
    ama_weights::{self, Loaded, IMPLEMENTATION_COMMIT},
    ama_exchange::{self, Branch, Rules},
    ama_f::{self, Outcome, Plan, Policy},
    ama_search::{self, Config, Pair},
    bitboard::{
        board_from_grid, board_to_grid, BitBoard, BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA,
        COLOR_PURPLE, COLOR_RED,
    },
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

const F_MAX_BUDGET_MS: f64 = 120000.;
const MS_PER_SECOND: f64 = 1000.;

fn config(width: usize, depth: usize, colors: Vec<u8>, loaded: &Loaded, form: bool) -> PyResult<Config> {
    let mut weight = loaded.weight;
    if !form {
        weight.form = 0;
    }
    let colors = colors.try_into().map_err(|_| PyValueError::new_err("colorsは4要素です"))?;
    Ok(Config { trigger: ama_search::DEFAULT_TRIGGER, width, depth, colors, weight })
}

fn provenance(py: Python<'_>, loaded: &Loaded, commit: Option<&str>) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("ama_commit", commit.unwrap_or(IMPLEMENTATION_COMMIT))?;
    d.set_item("implementation_commit", IMPLEMENTATION_COMMIT)?;
    d.set_item("weights_sha256", &loaded.sha256)?;
    Ok(d.into_py(py))
}

#[pyfunction]
#[pyo3(signature=(weights_path=None,ama_commit=None))]
fn ama_version_py(py: Python<'_>, weights_path: Option<&str>, ama_commit: Option<&str>) -> PyResult<PyObject> {
    let loaded = ama_weights::load(weights_path, "build").map_err(PyValueError::new_err)?;
    provenance(py, &loaded, ama_commit)
}

fn board(grid: Vec<u8>, tape: &[Pair], cfg: &Config) -> PyResult<BitBoard> {
    #[cfg(feature = "ama-profile")]
    let _timer = crate::ama_profile_clock::Timer::new(7);
    if grid.len() != BOARD_ROWS * BOARD_COLS
        || grid
            .iter()
            .any(|&v| v != COLOR_EMPTY && v != COLOR_OJAMA && !(COLOR_RED..=COLOR_PURPLE).contains(&v))
    {
        return Err(PyValueError::new_err("盤面は78要素、色は0/1..5/9（UNKNOWN不可）です"));
    }
    let b = board_from_grid(&grid);
    ama_f::validate(&b, tape, cfg).map_err(PyValueError::new_err)?;
    Ok(b)
}

fn check_counts(k: usize, incoming: i32, target: i32) -> PyResult<()> {
    if k > ama_f::MAX_HANDS
        || !(0..=ama_f::MAX_REQUEST_VALUE).contains(&incoming)
        || !(0..=ama_f::MAX_REQUEST_VALUE).contains(&target)
    {
        return Err(PyValueError::new_err("Kは0..128、予告/目標得点は0..1000000です"));
    }
    Ok(())
}

fn outcome(py: Python<'_>, out: &Outcome) -> PyResult<PyObject> {
    #[cfg(feature = "ama-profile")]
    let _timer = crate::ama_profile_clock::Timer::new(8);
    let d = PyDict::new_bound(py);
    d.set_item("fired", out.fired)?;
    d.set_item("target_reached", out.reached)?;
    d.set_item("fire_hand", out.fire_hand)?;
    d.set_item("score", out.score)?;
    d.set_item("max_score", out.max_score)?;
    d.set_item("total_score", out.total_score)?;
    d.set_item("carry_in", out.carry_in)?;
    d.set_item("score_remainder", out.score_remainder)?;
    d.set_item("cancelled", out.cancelled)?;
    d.set_item("excess", out.excess)?;
    d.set_item("fallen", out.fallen)?;
    d.set_item("pending", out.pending)?;
    d.set_item("overflow", out.overflow)?;
    d.set_item("dead", out.dead)?;
    d.set_item("coverage", out.coverage)?;
    if out.interrupted { d.set_item("truncated", true)?; }
    d.set_item("final_board", board_to_grid(&out.board))?;
    d.set_item(
        "post_fire_board",
        out.steps.iter().rev().find(|s| s.score > 0).map(|s| board_to_grid(&s.board)),
    )?;
    let steps = out
        .steps
        .iter()
        .map(|s| (s.hand, s.placement.0, s.placement.1, s.score, s.chain, board_to_grid(&s.board)))
        .collect::<Vec<_>>();
    d.set_item("steps", steps)?;
    Ok(d.into_py(py))
}

fn branches(py: Python<'_>, nodes: &[Branch]) -> PyResult<Vec<PyObject>> {
    nodes
        .iter()
        .map(|n| {
            let d = PyDict::new_bound(py);
            d.set_item("side", n.side)?;
            if let Some(ms) = n.completed_ms {
                d.set_item("completed_ms", ms)?;
                d.set_item("progress_truncated", n.progress_truncated)?;
            }
            d.set_item("level", n.level)?;
            d.set_item("policy", match n.plan.policy { Policy::Main => "main", Policy::Counter => "counter", Policy::QuickCounter => "quick_counter" })?;
            if n.unavailable { d.set_item("unavailable", true)?; }
            d.set_item("deadline_hands", n.plan.hands)?;
            d.set_item("target_score", n.plan.target)?;
            d.set_item("start_index", n.start_index)?;
            d.set_item("end_index", n.end_index)?;
            d.set_item("start_ms", n.start_ms)?;
            d.set_item("fire_ms", n.fire_ms)?;
            d.set_item("end_ms", n.end_ms)?;
            d.set_item("available_hands", n.available_hands)?;
            d.set_item("balance", n.balance)?;
            d.set_item("landed", n.landed.to_vec())?;
            d.set_item("pending", n.pending.to_vec())?;
            d.set_item("truncated", n.truncated)?;
            d.set_item("outcome", outcome(py, &n.outcome)?)?;
            d.set_item("children", branches(py, &n.children)?)?;
            if let Some(p) = &n.preparation { d.set_item("preparation", preparation(py, p)?)?; }
            Ok(d.into_py(py))
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature=(grid,pairs,k,width,depth,colors,policy="main",incoming=0,target_score=0,profile="build",form=true,stop=true,weights_path=None,ama_commit=None,opponent_consumption=None,quick_visible=2,quick_min_return=9,budget_ms=None))]
fn ama_f_py(
    py: Python<'_>,
    grid: Vec<u8>,
    pairs: Vec<Pair>,
    k: usize,
    width: usize,
    depth: usize,
    colors: Vec<u8>,
    policy: &str,
    incoming: i32,
    target_score: i32,
    profile: &str,
    form: bool,
    stop: bool,
    weights_path: Option<&str>,
    ama_commit: Option<&str>,
    opponent_consumption: Option<f64>, quick_visible: usize, quick_min_return: i32,
    budget_ms: Option<f64>,
) -> PyResult<PyObject> {
    let deadline = f_deadline(budget_ms)?;
    #[cfg(feature = "ama-profile")]
    crate::ama_profile_clock::take();
    check_counts(k, incoming, target_score)?;
    let loaded = ama_weights::load(weights_path, profile).map_err(PyValueError::new_err)?;
    let cfg = config(width, depth, colors, &loaded, form)?;
    let board = board(grid, &pairs, &cfg)?;
    if policy == "quick_counter" {
        let result = quick_result(py, board, &pairs, k, &cfg, incoming, target_score,
            opponent_consumption, quick_visible, quick_min_return, deadline)?;
        result.bind(py).downcast::<PyDict>()?.set_item("provenance", provenance(py, &loaded, ama_commit)?)?;
        return Ok(result);
    }
    let policy = match policy {
        "main" => Policy::Main,
        "counter" => Policy::Counter,
        _ => return Err(PyValueError::new_err("policyはmain/counter/quick_counterです")),
    };
    let target = if policy == Policy::Counter && target_score == 0 {
        incoming * ama_f::OJAMA_RATE
    } else {
        target_score
    };
    let mut out = py
        .allow_threads(|| crate::ama_budget::within(deadline, ||
            ama_f::run(board, &pairs, Plan { policy, hands: k, target }, &cfg, incoming, stop)));
    ama_f::settle(&mut out, &[0, 1, 2, 3, 4, 5]);
    let result = outcome(py, &out)?;
    result.bind(py).downcast::<PyDict>()?.set_item("provenance", provenance(py, &loaded, ama_commit)?)?;
    #[cfg(feature = "ama-profile")]
    result.bind(py).downcast::<PyDict>()?.set_item("_profile", crate::ama_profile_clock::take().to_vec())?;
    Ok(result)
}

// 明示指定時だけ期限を設ける。既存呼出しは従来どおり無期限。
fn f_deadline(budget_ms: Option<f64>) -> PyResult<Option<std::time::Instant>> {
    match budget_ms {
        Some(ms) if !ms.is_finite() || ms < 0. || ms > F_MAX_BUDGET_MS =>
            Err(PyValueError::new_err("予算は0..120000msの有限値です")),
        Some(ms) => Ok(Some(std::time::Instant::now()+std::time::Duration::from_secs_f64(ms/MS_PER_SECOND))),
        None => Ok(None),
    }
}

#[pyfunction]
#[pyo3(signature=(grid,pairs,width,depth,colors,profile="build",form=true,cap=0,weights_path=None,ama_commit=None,with_metadata=false))]
fn ama_search_py(
    py: Python<'_>,
    grid: Vec<u8>,
    pairs: Vec<Pair>,
    width: usize,
    depth: usize,
    colors: Vec<u8>,
    profile: &str,
    form: bool,
    cap: i32,
    weights_path: Option<&str>,
    ama_commit: Option<&str>,
    with_metadata: bool,
) -> PyResult<PyObject> {
    let loaded = ama_weights::load(weights_path, profile).map_err(PyValueError::new_err)?;
    let cfg = config(width, depth, colors, &loaded, form)?;
    let b = board(grid, &pairs, &cfg)?;
    if pairs.len() < 2 {
        return Err(PyValueError::new_err("先読みには最低2組が必要です"));
    }
    let rows: Vec<_> = py
        .allow_threads(|| ama_search::multi(b, &pairs, depth, &cfg, cap))
        .iter()
        .map(|c| (c.placement.0, c.placement.1, c.score))
        .collect();
    if !with_metadata { return Ok(rows.into_py(py)); }
    let d = PyDict::new_bound(py);
    d.set_item("candidates", rows)?;
    d.set_item("provenance", provenance(py, &loaded, ama_commit)?)?;
    Ok(d.into_py(py))
}

#[pyfunction]
#[pyo3(signature=(own,opponent,common_pairs,indices,k,width,depth,colors,levels,hand_ms,chain_ms,main_targets,deadlines,incoming=0,profile="build",form=true,both=false,parallel=false,garbage_order=None,weights_path=None,ama_commit=None,budget_ms=None,candidate_limit=None,level_widths=None,num_threads=None,advance_opponent=false,preparation_mode="shared",opponent_horizon=None,opponent_target=0,quick_candidate=false,quick_visible=2,quick_min_return=9,opponent_busy_ms=0,budget_clock_scale=None,record_progress=false))]
fn ama_exchange_py(
    py: Python<'_>,
    own: Vec<u8>, opponent: Vec<u8>,
    common_pairs: Vec<Pair>, indices: (usize, usize),
    k: usize, width: usize, depth: usize,
    colors: Vec<u8>, levels: usize, hand_ms: u64,
    chain_ms: Vec<u64>,
    main_targets: Vec<i32>,
    deadlines: Vec<usize>,
    incoming: i32, profile: &str, form: bool, both: bool, parallel: bool,
    garbage_order: Option<Vec<usize>>,
    weights_path: Option<&str>,
    ama_commit: Option<&str>,
    budget_ms: Option<u64>,
    candidate_limit: Option<usize>,
    level_widths: Option<Vec<usize>>,
    num_threads: Option<usize>,
    advance_opponent: bool, preparation_mode: &str, opponent_horizon: Option<usize>, opponent_target: i32,
    quick_candidate: bool, quick_visible: usize, quick_min_return: i32,
    opponent_busy_ms: u64,
    budget_clock_scale: Option<f64>,
    record_progress: bool,
) -> PyResult<PyObject> {
    let progress_start = record_progress.then(std::time::Instant::now);
    check_counts(k, incoming, 0)?;
    let loaded = ama_weights::load(weights_path, profile).map_err(PyValueError::new_err)?;
    let cfg = config(width, depth, colors, &loaded, form)?;
    let boards = [board(own, &common_pairs, &cfg)?, board(opponent, &common_pairs, &cfg)?];
    let indices = [indices.0, indices.1];
    let order = garbage_order
        .unwrap_or_else(|| vec![0, 1, 2, 3, 4, 5])
        .try_into()
        .map_err(|_| PyValueError::new_err("落下順は6要素です"))?;
    let rules = Rules { levels, hand_ms, chain_ms, main_targets, deadlines, garbage_order: order };
    ama_exchange::validate(&rules, indices, &common_pairs).map_err(PyValueError::new_err)?;
    if both && opponent_busy_ms > 0 {
        return Err(PyValueError::new_err("初期連鎖待ちはboth=falseで方向別に指定してください"));
    }
    let opt = ama_exchange::anytime::Options { budget_ms, budget_clock_scale, progress_start, candidate_limit, opponent_busy_ms,
        advance_opponent, preparation_mode: preparation_mode.into(), opponent_horizon, opponent_target,
        quick_candidate, quick_visible, quick_min_return,
        widths: level_widths.unwrap_or_default(), threads: num_threads };
    opt.validate().map_err(PyValueError::new_err)?;
    let (a, b) = py.allow_threads(|| ama_exchange::anytime::directions(boards, &common_pairs,
        indices, k, incoming, &cfg, &rules, &opt, both, parallel)).map_err(PyValueError::new_err)?;
    let d = PyDict::new_bound(py);
    if let Some(start) = progress_start {
        d.set_item("progress_elapsed_ms", start.elapsed().as_secs_f64() * 1000.0)?;
    }
    d.set_item("own", branches(py, &a)?)?;
    d.set_item("opponent", branches(py, &b)?)?;
    d.set_item("board_basis", if advance_opponent { "simultaneous_initial_boards" } else { "opponent_at_root_fire" })?;
    d.set_item("provenance", provenance(py, &loaded, ama_commit)?)?;
    if opt.enabled() {
        d.set_item("budget_exhausted", a.iter().chain(&b).any(ama_exchange::anytime::interrupted))?;
    }
    Ok(d.into_py(py))
}

fn preparation(py: Python<'_>, p: &ama_exchange::prepare::Preparation) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("mode", &p.mode)?;
    d.set_item("horizon", p.horizon)?;
    d.set_item("compute_ms", p.compute_ms)?;
    d.set_item("reused", p.reused)?;
    d.set_item("start_index", p.start_index)?;
    d.set_item("end_index", p.end_index)?;
    d.set_item("fire_ms", p.fire_ms)?;
    d.set_item("end_ms", p.end_ms)?;
    d.set_item("board", board_to_grid(&p.board))?;
    d.set_item("score", p.score)?;
    d.set_item("sent", p.score / ama_f::OJAMA_RATE)?;
    d.set_item("score_remainder", p.score % ama_f::OJAMA_RATE)?;
    d.set_item("unsupported", p.unsupported)?;
    Ok(d.into_py(py))
}

fn quick_result(py: Python<'_>, board: BitBoard, tape: &[Pair], k: usize, cfg: &Config,
                incoming: i32, target: i32, consumption: Option<f64>, visible: usize,
                threshold: i32, deadline: Option<std::time::Instant>) -> PyResult<PyObject> {
    if !(2..=3).contains(&visible) || !(1..=ama_f::MAX_REQUEST_VALUE).contains(&threshold)
        || consumption.is_some_and(|r| !r.is_finite() || !(0.0..=1.0).contains(&r)) {
        return Err(PyValueError::new_err("可視手は2/3、返却閾値は正の個数、消費率は0..1です"));
    }
    let assumptions = consumption.map_or(vec![("provocation", 0.0), ("main", 1.0)], |r|
        vec![(if r < ama_f::quick::CONSUMPTION_LIMIT { "provocation" } else { "main" }, r)]);
    let mut results = Vec::new();
    let mut main_cache: Option<Outcome> = None;
    for (assumption, ratio) in assumptions {
        let plan = Plan { policy: Policy::Main, hands: k, target };
        let (mut out, probe, counter) = if let Some(out) = main_cache.take() {
            (out, ama_f::quick::Probe { best: None, maximum_return: 0, expanded: 0, complete: true }, false)
        } else { py.allow_threads(|| crate::ama_budget::within(deadline, ||
            ama_f::quick::run(board, tape, plan, cfg, incoming, ratio, visible, threshold))) };
        if !counter { main_cache = Some(out.clone()); }
        ama_f::settle(&mut out, &[0, 1, 2, 3, 4, 5]);
        let item = outcome(py, &out)?;
        let d = item.bind(py).downcast::<PyDict>()?;
        d.set_item("assumption", assumption)?;
        d.set_item("selected_policy", if counter { "counter" } else { "main" })?;
        d.set_item("quick_complete", probe.complete)?;
        d.set_item("quick_max_return", probe.maximum_return)?;
        d.set_item("quick_expanded", probe.expanded)?;
        d.set_item("quick_visible", visible)?;
        d.set_item("quick_min_return", threshold)?;
        results.push(item);
    }
    if consumption.is_some() { return Ok(results.remove(0)); }
    let d = PyDict::new_bound(py);
    d.set_item("opponent_consumption", py.None())?;
    d.set_item("assumptions", results)?;
    Ok(d.into_py(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ama_race_detail_py, m)?)?;
    m.add_function(wrap_pyfunction!(ama_race_curve_py, m)?)?;
    m.add_function(wrap_pyfunction!(ama_branches_py, m)?)?;
    m.add_function(wrap_pyfunction!(ama_growth_py, m)?)?;
    m.add_function(wrap_pyfunction!(ama_f_py, m)?)?;
    m.add_function(wrap_pyfunction!(ama_search_py, m)?)?;
    m.add_function(wrap_pyfunction!(ama_exchange_py, m)?)?;
    m.add_function(wrap_pyfunction!(ama_version_py, m)?)?;
    Ok(())
}

/// 予告を含む競争実験。時間期限は受け取らない。
#[pyfunction]
#[pyo3(signature=(grid,pairs,horizon,colors,width=16,depth=8,visible_pairs=3,pending=0))]
fn ama_race_detail_py(py: Python<'_>, grid: Vec<u8>, pairs: Vec<Pair>, horizon: usize,
                     colors: Vec<u8>, width: usize, depth: usize, visible_pairs: usize,
                     pending: i32) -> PyResult<(Vec<ama_f::race_detail::Fire>, Vec<ama_f::race_detail::State>)> {
    check_counts(horizon, pending, 0)?;
    if !(2..=3).contains(&visible_pairs) { return Err(PyValueError::new_err("可視組数は2または3です")); }
    let loaded = ama_weights::load(None, "build").map_err(PyValueError::new_err)?;
    let cfg = config(width, depth, colors, &loaded, true)?;
    let b = board(grid, &pairs, &cfg)?;
    Ok(py.allow_threads(|| crate::ama_budget::within(None, ||
        ama_f::race_detail::curve(b, &pairs, horizon, &cfg, visible_pairs, pending))))
}

/// 終了盤を必要とする競争実験だけの入口。本番への接続と時間打切りはない。
#[pyfunction]
#[pyo3(signature=(grid,pairs,horizon,colors,width=16,depth=8,visible_pairs=3))]
fn ama_race_curve_py(py: Python<'_>, grid: Vec<u8>, pairs: Vec<Pair>, horizon: usize,
                     colors: Vec<u8>, width: usize, depth: usize,
                     visible_pairs: usize) -> PyResult<Vec<ama_f::race::Point>> {
    check_counts(horizon, 0, 0)?;
    if !(2..=3).contains(&visible_pairs) {
        return Err(PyValueError::new_err("可視組数は2または3です"));
    }
    let loaded = ama_weights::load(None, "build").map_err(PyValueError::new_err)?;
    let cfg = config(width, depth, colors, &loaded, true)?;
    let b = board(grid, &pairs, &cfg)?;
    Ok(py.allow_threads(|| crate::ama_budget::within(None, ||
        ama_f::race::curve(b, &pairs, horizon, &cfg, visible_pairs))))
}

/// v4.4の明示呼出し専用。旧APIと本番の探索は変更しない。
#[pyfunction]
#[pyo3(signature=(grid,pairs,horizon,colors,width=1,depth=3,budget_ms=None,visible_pairs=3))]
fn ama_branches_py(py: Python<'_>, grid: Vec<u8>, pairs: Vec<Pair>, horizon: usize,
                   colors: Vec<u8>, width: usize, depth: usize,
                   budget_ms: Option<f64>, visible_pairs: usize) -> PyResult<Vec<ama_f::branches::Point>> {
    check_counts(horizon, 0, 0)?;
    if !(2..=3).contains(&visible_pairs) {
        return Err(PyValueError::new_err("可視組数は2または3です"));
    }
    let deadline = f_deadline(budget_ms)?;
    let loaded = ama_weights::load(None, "build").map_err(PyValueError::new_err)?;
    let cfg = config(width, depth, colors, &loaded, true)?;
    let b = board(grid, &pairs, &cfg)?;
    Ok(py.allow_threads(|| crate::ama_budget::within(deadline, ||
        ama_f::branches::curve(b, &pairs, horizon, &cfg, visible_pairs))))
}

/// 明示呼出専用。既存F、本番探索、既定値は変更しない。
#[pyfunction]
#[pyo3(signature=(grid,pairs,horizon,colors,width=1,depth=3,budget_ms=None,visible_pairs=3))]
fn ama_growth_py(py: Python<'_>, grid: Vec<u8>, pairs: Vec<Pair>, horizon: usize,
                 colors: Vec<u8>, width: usize, depth: usize,
                 budget_ms: Option<f64>, visible_pairs: usize) -> PyResult<Vec<ama_f::growth::Point>> {
    check_counts(horizon, 0, 0)?;
    if !(2..=3).contains(&visible_pairs) {
        return Err(PyValueError::new_err("可視組数は2または3です"));
    }
    let deadline = f_deadline(budget_ms)?;
    let loaded = ama_weights::load(None, "build").map_err(PyValueError::new_err)?;
    let cfg = config(width, depth, colors, &loaded, true)?;
    let b = board(grid, &pairs, &cfg)?;
    Ok(py.allow_threads(|| crate::ama_budget::within(deadline, ||
        ama_f::growth::curve(b, &pairs, horizon, &cfg, visible_pairs))))
}

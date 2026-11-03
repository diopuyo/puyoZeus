//! 正の ama の JSON 入口。PythonのGILを解放して独立Rustモジュールを呼ぶ。
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyfunction]
#[pyo3(signature = (request_json, threads=2))]
fn ama_reference_json_py(py: Python<'_>, request_json: String, threads: usize) -> PyResult<String> {
    py.allow_threads(move || {
        let input = serde_json::from_str(&request_json).map_err(|e| e.to_string())?;
        let output = ama_reference::api::request_with_threads(&input, threads)?;
        serde_json::to_string(&output).map_err(|e| e.to_string())
    }).map_err(PyValueError::new_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ama_reference_json_py, m)?)
}

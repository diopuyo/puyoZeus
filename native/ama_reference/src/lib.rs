//! 配信用 ama の独立実装。旧方式 F の API からは呼び出さない。
mod bits;
pub mod field;
pub mod chain;
mod form_data;
pub mod form;
pub mod eval;
pub mod order;
mod table;
pub mod search;
pub mod api;

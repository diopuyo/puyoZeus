//! 呼出単位の協調打切り。各評価スレッドに同じ絶対期限を設定する。
use std::cell::Cell;
use std::time::Instant;

thread_local! { static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) }; }

pub fn expired() -> bool {
    DEADLINE.with(|value| value.get().is_some_and(|t| Instant::now() >= t))
}

#[cfg(feature = "ama-reuse-identical-queues")]
pub fn is_unbounded() -> bool {
    DEADLINE.with(|value| value.get().is_none())
}

pub fn within<T>(deadline: Option<Instant>, action: impl FnOnce() -> T) -> T {
    struct Reset(Option<Instant>);
    impl Drop for Reset {
        fn drop(&mut self) { DEADLINE.with(|value| value.set(self.0)); }
    }
    let _reset = Reset(DEADLINE.with(|value| value.replace(deadline)));
    action()
}

//! 明示ビルド時だけ有効な単体ベンチ用工程時計。
use std::cell::RefCell;
use std::time::Instant;
const STAGES: usize = 10;
thread_local! {
    static TOTAL: RefCell<[(u64, u64); STAGES]> = const { RefCell::new([(0, 0); STAGES]) };
}
pub struct Timer(usize, Instant);
impl Timer {
    pub fn new(stage: usize) -> Self { Self(stage, Instant::now()) }
}
impl Drop for Timer {
    fn drop(&mut self) {
        TOTAL.with(|total| {
            let mut total = total.borrow_mut();
            total[self.0].0 += 1;
            total[self.0].1 += self.1.elapsed().as_nanos() as u64;
        });
    }
}
pub fn take() -> [(u64, u64); STAGES] {
    TOTAL.with(|total| std::mem::replace(&mut *total.borrow_mut(), [(0, 0); STAGES]))
}

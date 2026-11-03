//! 基準 exe の libstdc++ introsort の同点順序を再現する。
//! 比較は厳密な less。候補番号等の副キーを追加しない。
const THRESHOLD: usize = 16;

fn pivot<T, F: Fn(&T, &T) -> bool>(a: &mut [T], less: &F) {
    let (b, c, d) = (1, a.len() / 2, a.len() - 1);
    let median = if less(&a[b], &a[c]) {
        if less(&a[c], &a[d]) { c } else if less(&a[b], &a[d]) { d } else { b }
    } else if less(&a[b], &a[d]) { b } else if less(&a[c], &a[d]) { d } else { c };
    a.swap(0, median);
}

fn partition<T, F: Fn(&T, &T) -> bool>(a: &mut [T], less: &F) -> usize {
    pivot(a, less);
    let (mut lo, mut hi) = (1, a.len());
    loop {
        while less(&a[lo], &a[0]) { lo += 1; }
        hi -= 1;
        while less(&a[0], &a[hi]) { hi -= 1; }
        if lo >= hi { return lo; }
        a.swap(lo, hi); lo += 1;
    }
}

fn adjust_heap<T: Clone, F: Fn(&T, &T) -> bool>(a: &mut [T], start: usize, value: T, less: &F) {
    let (top, mut hole, mut child) = (start, start, start);
    while child < (a.len() - 1) / 2 {
        child = 2 * (child + 1);
        if less(&a[child], &a[child - 1]) { child -= 1; }
        a[hole] = a[child].clone(); hole = child;
    }
    if a.len() % 2 == 0 && child == (a.len() - 2) / 2 {
        child = 2 * (child + 1);
        a[hole] = a[child - 1].clone(); hole = child - 1;
    }
    while hole > top {
        let parent = (hole - 1) / 2;
        if !less(&a[parent], &value) { break; }
        a[hole] = a[parent].clone(); hole = parent;
    }
    a[hole] = value;
}

fn heap_sort<T: Clone, F: Fn(&T, &T) -> bool>(a: &mut [T], less: &F) {
    for parent in (0..a.len() / 2).rev() {
        adjust_heap(a, parent, a[parent].clone(), less);
    }
    for end in (1..a.len()).rev() {
        let value = a[end].clone();
        a[end] = a[0].clone();
        adjust_heap(&mut a[..end], 0, value, less);
    }
}

fn introsort<T: Clone, F: Fn(&T, &T) -> bool>(mut a: &mut [T], mut depth: usize, less: &F) {
    while a.len() > THRESHOLD {
        if depth == 0 { heap_sort(a, less); return; }
        depth -= 1;
        let cut = partition(a, less);
        let (left, right) = a.split_at_mut(cut);
        introsort(right, depth, less);
        a = left;
    }
}

pub fn sort<T: Clone, F: Fn(&T, &T) -> bool>(a: &mut [T], less: F) {
    if a.is_empty() { return; }
    let depth = (usize::BITS - 1 - a.len().leading_zeros()) as usize * 2;
    introsort(a, depth, &less);
    // introsort後は先頭区間に最小値があり、全体の挿入整列で同じ移動順になる。
    for i in 1..a.len() {
        let value = a[i].clone();
        let mut hole = i;
        while hole > 0 && less(&value, &a[hole - 1]) {
            a[hole] = a[hole - 1].clone(); hole -= 1;
        }
        a[hole] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sort_handles_equal_and_reversed() {
        for n in 0..260 {
            let mut values: Vec<usize> = (0..n).rev().map(|i| i % 7).collect();
            sort(&mut values, |a, b| a < b);
            assert!(values.windows(2).all(|v| v[0] <= v[1]));
        }
    }
}

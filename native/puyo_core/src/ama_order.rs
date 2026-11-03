//! オラクルの libc++ ヒープ順序を再現する（同点の子は左優先）。
//! LLVM libcxx __algorithm/{make_heap,pop_heap,push_heap,sift_down,sort_heap}.h 由来。
//! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception（licenses/LLVM.txt）。

pub fn push<T: Copy>(a: &mut [T], key: impl Fn(T) -> i32) {
    if a.is_empty() {
        return;
    }
    let mut i = a.len() - 1;
    let value = a[i];
    while i > 0 {
        let parent = (i - 1) / 2;
        if key(a[parent]) <= key(value) {
            break;
        }
        a[i] = a[parent];
        i = parent;
    }
    a[i] = value;
}

pub fn pop<T: Copy>(a: &mut [T], key: impl Fn(T) -> i32 + Copy) {
    let n = a.len();
    if n < 2 {
        return;
    }
    let top = a[0];
    let mut hole = 0;
    while hole <= (n - 2) / 2 {
        let mut child = hole * 2 + 1;
        if child + 1 < n && key(a[child]) > key(a[child + 1]) {
            child += 1;
        }
        a[hole] = a[child];
        hole = child;
    }
    if hole == n - 1 {
        a[hole] = top;
    } else {
        a[hole] = a[n - 1];
        a[n - 1] = top;
        push(&mut a[..hole + 1], key);
    }
}

pub fn make<T: Copy>(a: &mut [T], key: impl Fn(T) -> i32 + Copy) {
    let n = a.len();
    if n < 2 {
        return;
    }
    for start in (0..=(n - 2) / 2).rev() {
        let top = a[start];
        let mut hole = start;
        while hole <= (n - 2) / 2 {
            let mut child = hole * 2 + 1;
            if child + 1 < n && key(a[child]) > key(a[child + 1]) {
                child += 1;
            }
            if key(a[child]) > key(top) {
                break;
            }
            a[hole] = a[child];
            hole = child;
        }
        a[hole] = top;
    }
}

pub fn sort_heap<T: Copy>(a: &mut [T], key: impl Fn(T) -> i32 + Copy) {
    for end in (2..=a.len()).rev() {
        pop(&mut a[..end], key);
    }
}

pub fn insertion<T: Copy>(a: &mut [T], key: impl Fn(T) -> i32) {
    // 候補数は最大22。オラクルの libc++ std::sort は24未満で挿入ソート。
    for i in 1..a.len() {
        let value = a[i];
        let mut j = i;
        while j > 0 && key(value) > key(a[j - 1]) {
            a[j] = a[j - 1];
            j -= 1;
        }
        a[j] = value;
    }
}

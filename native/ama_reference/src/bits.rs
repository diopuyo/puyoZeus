//! 盤のビット列 (128 bit、列 16 bit) の演算。x86_64 では SSE2 (全 64 bit CPU にある) の 1 命令で行う。
//! 列の中の上下のずらし (shr1/shl1) は列をまたがない。列 16 bit の最下位 (壁の段) と最上位は
//! 常に 0 なので、u128 のずらしと結果は同じ。
use std::ops::{BitAnd, BitOr, BitXor};

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub struct V(pub __m128i);

    impl V {
        #[inline(always)] pub fn new(x: u128) -> Self { V(unsafe { core::mem::transmute::<u128, __m128i>(x) }) }
        #[inline(always)] pub fn get(self) -> u128 { unsafe { core::mem::transmute::<__m128i, u128>(self.0) } }
        #[inline(always)] pub fn and(self, o: V) -> V { V(unsafe { _mm_and_si128(self.0, o.0) }) }
        #[inline(always)] pub fn or(self, o: V) -> V { V(unsafe { _mm_or_si128(self.0, o.0) }) }
        #[inline(always)] pub fn xor(self, o: V) -> V { V(unsafe { _mm_xor_si128(self.0, o.0) }) }
        /// self & !o
        #[inline(always)] pub fn andnot(self, o: V) -> V { V(unsafe { _mm_andnot_si128(o.0, self.0) }) }
        /// 1 段上のセルの値を下へ (u128 の >> 1)。
        #[inline(always)] pub fn up(self) -> V { V(unsafe { _mm_srli_epi16::<1>(self.0) }) }
        /// 1 段下のセルの値を上へ (u128 の << 1)。
        #[inline(always)] pub fn down(self) -> V { V(unsafe { _mm_slli_epi16::<1>(self.0) }) }
        /// 左の列の値を右へ (u128 の << 16)。
        #[inline(always)] pub fn left(self) -> V { V(unsafe { _mm_slli_si128::<2>(self.0) }) }
        /// 右の列の値を左へ (u128 の >> 16)。
        #[inline(always)] pub fn right(self) -> V { V(unsafe { _mm_srli_si128::<2>(self.0) }) }
        #[inline(always)] pub fn same(self, o: V) -> bool {
            unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(self.0, o.0)) == 0xFFFF }
        }
        #[inline(always)] pub fn is_zero(self) -> bool { self.same(V(unsafe { _mm_setzero_si128() })) }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod imp {
    #[derive(Clone, Copy)]
    pub struct V(pub u128);

    impl V {
        #[inline(always)] pub fn new(x: u128) -> Self { V(x) }
        #[inline(always)] pub fn get(self) -> u128 { self.0 }
        #[inline(always)] pub fn and(self, o: V) -> V { V(self.0 & o.0) }
        #[inline(always)] pub fn or(self, o: V) -> V { V(self.0 | o.0) }
        #[inline(always)] pub fn xor(self, o: V) -> V { V(self.0 ^ o.0) }
        #[inline(always)] pub fn andnot(self, o: V) -> V { V(self.0 & !o.0) }
        #[inline(always)] pub fn up(self) -> V { V(self.0 >> 1) }
        #[inline(always)] pub fn down(self) -> V { V(self.0 << 1) }
        #[inline(always)] pub fn left(self) -> V { V(self.0 << 16) }
        #[inline(always)] pub fn right(self) -> V { V(self.0 >> 16) }
        #[inline(always)] pub fn same(self, o: V) -> bool { self.0 == o.0 }
        #[inline(always)] pub fn is_zero(self) -> bool { self.0 == 0 }
    }
}

pub use imp::V;

impl BitAnd for V { type Output = V; #[inline(always)] fn bitand(self, o: V) -> V { self.and(o) } }
impl BitOr for V { type Output = V; #[inline(always)] fn bitor(self, o: V) -> V { self.or(o) } }
impl BitXor for V { type Output = V; #[inline(always)] fn bitxor(self, o: V) -> V { self.xor(o) } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::CELLS;

    #[test]
    fn shifts_match_u128_on_cell_bits() {
        let mut state = 0x1234_5678_9abc_def1u64;
        for _ in 0..10_000 {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let a = ((state as u128) << 64 | (state.rotate_left(17) as u128)) & CELLS;
            let v = V::new(a);
            assert_eq!(v.up().get(), a >> 1);
            assert_eq!(v.down().get(), a << 1);
            assert_eq!(v.left().get(), a << 16);
            assert_eq!(v.right().get(), a >> 16);
            assert!(v.same(V::new(a)));
            assert_eq!(v.is_zero(), a == 0);
        }
    }
}

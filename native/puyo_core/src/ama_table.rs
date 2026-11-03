//! ama ai/search/beam/{table.cpp,node.h} (MIT, citrus610, 2023) の移植。
//! rapidhash は Nicolas De Carli, 2024, BSD-2-Clause（licenses/rapidhash-BSD.txt）。
use crate::bitboard::BitBoard;

const BUCKETS: usize = 8192;
const ENTRIES: usize = 4;
const RAPID_SEED: u64 = 0xbdd89aa982704029;
const SECRET: [u64; 3] = [0x2d358dccaa6c78a5, 0x8bb84b93962eacc9, 0x4b33a62ed433d4a3];
const HASH_BYTES: u64 = 80;

fn mix(a: u64, b: u64) -> u64 {
    let product = a as u128 * b as u128;
    product as u64 ^ (product >> 64) as u64
}

pub fn hash(b: &BitBoard, colors: &[u8; 4]) -> u64 {
    // ama の FieldBit は各色につき u16×8、末尾2列はゼロ。
    let mut words = [0u64; 10];
    for i in 0..5 {
        let p = if i < 4 { &b.colors[colors[i] as usize - 1] } else { &b.ojama };
        words[i * 2] = (0..4).fold(0, |a, x| a | ((p[x] as u64) << (16 * x)));
        words[i * 2 + 1] = p[4] as u64 | ((p[5] as u64) << 16);
    }
    let initial = RAPID_SEED ^ mix(RAPID_SEED ^ SECRET[0], SECRET[1]) ^ HASH_BYTES;
    let mut seed = mix(words[0] ^ SECRET[0], words[1] ^ initial)
        ^ mix(words[2] ^ SECRET[1], words[3] ^ initial)
        ^ mix(words[4] ^ SECRET[2], words[5] ^ initial);
    seed = mix(words[6] ^ SECRET[2], words[7] ^ seed ^ SECRET[1]);
    let product = (words[8] ^ SECRET[1]) as u128 * (words[9] ^ seed) as u128;
    mix(product as u64 ^ SECRET[0] ^ HASH_BYTES, (product >> 64) as u64 ^ SECRET[1])
}

#[derive(Clone, Copy, Default)]
pub struct Entry {
    pub hash: u16,
    pub age: u16,
    pub action: i16,
    pub eval: i16,
}

pub struct Table {
    entries: Vec<[Entry; ENTRIES]>,
    age: u16,
}

impl Table {
    pub fn new() -> Self {
        #[cfg(feature = "ama-profile-detail")]
        let _timer = crate::ama_profile_clock::Timer::new(2);
        Self { entries: vec![[Entry::default(); ENTRIES]; BUCKETS], age: 0 }
    }
    pub fn update(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    pub fn probe(&mut self, hash: u64, action: i32, evaluate: impl FnOnce() -> i32) -> Option<i32> {
        let bucket = ((hash as u128 * BUCKETS as u128) >> 64) as usize;
        let entries = &mut self.entries[bucket];
        let found = entries.iter().position(|e| e.hash == hash as u16);
        let slot = found.unwrap_or_else(|| {
            let mut slot = 0;
            for i in 1..ENTRIES {
                let (a, b) = (entries[i], entries[slot]);
                if a.age < b.age
                    || (a.age == b.age && (a.action as i32 + a.eval as i32) < b.action as i32 + b.eval as i32)
                {
                    slot = i;
                }
            }
            slot
        });
        let entry = &mut entries[slot];
        // 原典は探索世代が違っても hash 一致を hit とみなす。
        if found.is_some() && action < entry.action as i32 {
            return None;
        }
        let eval = if found.is_some() { entry.eval as i32 } else { evaluate() };
        if entry.hash != hash as u16
            || entry.age != self.age
            || (entry.action as i32 + entry.eval as i32) < action + eval
        {
            *entry = Entry {
                hash: hash as u16,
                age: self.age,
                action: action as i16,
                eval: eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            };
        }
        Some(eval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_preserves_oracle_integer_storage() {
        let mut t = Table::new();
        let h = 123;
        assert_eq!(t.probe(h, -40000, || 50000), Some(50000));
        // 原典の action は i16 キャスト、eval は i16 範囲へ飽和する。
        assert_eq!(t.probe(h, 30000, || panic!("hitで再評価しない")), Some(32767));
        t.update();
        assert_eq!(t.probe(h, -1, || panic!("世代が違ってもhit")), None);
    }

    #[test]
    fn board_color_mapping_changes_hash() {
        let mut b = BitBoard::default();
        b.colors[0][0] = 7;
        assert_ne!(hash(&b, &[1, 2, 3, 4]), hash(&b, &[2, 1, 3, 4]));
        assert_eq!(std::mem::size_of::<Entry>() * ENTRIES * BUCKETS, 256 * 1024);
    }
}

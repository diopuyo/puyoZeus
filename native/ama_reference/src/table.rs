//! 正の置換表。世代の異なる hit、16bit切詰め、置換優先順を維持。
const BUCKETS: usize = 8192;
const WAYS: usize = 4;

#[derive(Clone, Copy, Default)]
pub struct Entry { hash: u16, age: u16, pub action: i16, pub eval: i32, occupied: bool }

pub struct Table { entries: Vec<[Entry; WAYS]>, age: u16 }

impl Default for Table {
    fn default() -> Self { Self { entries: vec![[Entry::default(); WAYS]; BUCKETS], age: 1 } }
}

impl Table {
    pub fn update(&mut self) {
        self.age = self.age.wrapping_add(1);
        if self.age == 0 { self.entries.fill([Entry::default(); WAYS]); self.age = 1; }
    }

    pub fn get(&self, hash: u64) -> (usize, usize, Option<Entry>) {
        let index = ((hash as u128 * BUCKETS as u128) >> 64) as usize;
        let bucket = &self.entries[index];
        if let Some(i) = bucket.iter().position(|e| e.occupied && e.hash == hash as u16) {
            return (index, i, Some(bucket[i]));
        }
        let mut replacement = 0;
        for (i, e) in bucket.iter().enumerate() {
            if !e.occupied { return (index, i, None); }
            let old = bucket[replacement];
            if e.age < old.age || (e.age == old.age && e.action as i32 + e.eval < old.action as i32 + old.eval) {
                replacement = i;
            }
        }
        (index, replacement, None)
    }

    pub fn set(&mut self, bucket: usize, slot: usize, hash: u64, action: i32, eval: i32) {
        let e = &mut self.entries[bucket][slot];
        if !e.occupied || e.hash != hash as u16 || e.age != self.age || e.action as i32 + e.eval < action + eval {
            *e = Entry { hash: hash as u16, age: self.age,
                action: action.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                eval: eval.clamp(i16::MIN as i32, i16::MAX as i32), occupied: true };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn older_generation_still_hits_and_clamps() {
        let mut t = Table::default();
        let (bucket, slot, hit) = t.get(123);
        assert!(hit.is_none());
        t.set(bucket, slot, 123, -40000, 50000);
        t.update();
        let hit = t.get(123).2.unwrap();
        assert_eq!(hit.action, i16::MIN);
        assert_eq!(hit.eval, i16::MAX as i32);
    }
}

use std::hash::Hash;

/// Invert a map by swapping keys and values
pub fn invert_map<K, V, MK, MV>(original: MK) -> MV
where
    K: Ord + Hash + Eq,
    V: Ord + Hash + Eq + Clone,
    MK: IntoIterator<Item = (K, V)>,
    MV: FromIterator<(V, K)>,
{
    original
        .into_iter()
        .map(|(key, value)| (value, key))
        .collect()
}

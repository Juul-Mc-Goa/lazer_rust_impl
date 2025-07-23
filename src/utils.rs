pub type Aes128Ctr64LE = ctr::Ctr64LE<aes::Aes128>;

/// Computes smallest power of 2 that is not smaller than `x`.
pub fn next_2_power(mut x: u64) -> u64 {
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x += 1;

    x
}

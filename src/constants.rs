pub const LOG_PRIME: u64 = 40;
pub const PRIME_OFFSET: u64 = 195;
/// The prime number `p` defining `A_p = Z / pZ`.
pub const PRIME: u64 = (1 << 40) - 195;
/// The inverse of `2` modulo `p`.
pub const ONE_HALF_MOD_PRIME: u64 = ((1 << 40) - 194) >> 1;

pub const PRIME_BYTES_LEN: usize = (LOG_PRIME as usize) >> 3;
pub const U128_LEN: usize = 128_usize.div_ceil(LOG_PRIME as usize);

/// the degree `d` defining `R_p = A_p[X] / (X^d + 1)`.
pub const DEGREE: u64 = 64;

#[allow(dead_code)]
pub const JL_MAX_NORM: u64 = 1 << 28;
pub const JL_MAX_NORM_SQ: u128 = 1 << 56;

/// Let `c` be a challenge polynomial. Then `l2(c * r) / l2(r) <=
/// CHALLENGE_NORM`, for all `r`.
pub const CHALLENGE_NORM: u64 = 14;
/// number of challenge polynomial coefficients that are `1` or `-1`.
pub const TAU1: u128 = 32;
/// number of challenge polynomial coefficients that are `2` or `-2`.
pub const TAU2: u128 = 8;

lazy_static! {
    /// the slack factor introduced when proving norm bounds
    pub static ref SLACK: f64 = (128_f64 / 30.0).sqrt();
    /// some factor used in the `sis_secure` function
    pub static ref LOG_DELTA: f64 = 1.00444_f64.log2();
}

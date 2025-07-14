/// the prime number `q` used throughout
pub const PRIME: u64 = (1 << 40) - 195;
#[allow(dead_code)]
pub const LOG_PRIME: u64 = 40;

/// the degree `d` defining `R_q = (Z/qZ)[X] / (X^d + 1)`
pub const DEGREE: u64 = 64;

lazy_static! {
    /// the slack factor introduced when proving norm bounds
    static ref SLACK: f64 = (128_f64 / 30.0).sqrt();
}

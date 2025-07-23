use crate::{
    constants::{LOG_PRIME, PRIME_BYTES_LEN},
    linear_algebra::{PolyVec, SparsePolyMatrix},
    ring::{BaseRingElem, PolyRingElem},
};

use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

/// A degree 2 polynomial in the `r` witness vectors s_1, ..., s_r in R_q^n:
///
/// sum(a_{i,j} <s_i, s_j>) + sum(<b_i, s_i>) + c
///
/// Here:
/// * a_{i,j} is in R_q
/// * b_i is in R_q^n
/// * c is in R_q
#[allow(dead_code)]
pub struct Constraint {
    pub degree: usize,
    pub quadratic_part: SparsePolyMatrix,
    pub linear_part: PolyVec,
    pub constant: PolyRingElem,
}

impl Constraint {
    pub fn new_raw(degree: usize) -> Self {
        Self {
            degree,
            quadratic_part: SparsePolyMatrix(Vec::new()),
            linear_part: PolyVec::new(),
            constant: PolyRingElem::zero(),
        }
    }
    pub fn new() -> Self {
        Self::new_raw(1)
    }
}

#[allow(dead_code)]
pub fn collaps_jl_proj_raw(
    constraint: &mut Constraint,
    r: usize,
    dim: usize,
    hash: &mut [u8; 16],
    projection: &[BaseRingElem; 256],
    jl_matrix: &Vec<u8>,
) {
    let mut hasher = Shake128::default();
    hasher.update(hash);
    let mut reader = hasher.finalize_xof();

    // REVIEW: hashbuf length
    // hashbuf: 32 bytes, then challenges (256 BaseRingElem), then ???
    let mut hashbuf = [0_u8; 32 + PRIME_BYTES_LEN * 256 + 24];
    reader.read(&mut hashbuf);
    // update hash
    hash.copy_from_slice(&hashbuf[..16]);

    let challenges = &hashbuf[32..];
    let mut alpha = [0_u64; 256];

    let build_u64 = |limbs: &[u8]| {
        let mut shift = 1 << 8_u64;
        let mut result = limbs[0] as u64;
        for i in 1..PRIME_BYTES_LEN {
            result += shift * limbs[i] as u64;
            shift *= 256;
        }

        result
    };

    // 256 = 32 * 8
    let primes_len = LOG_PRIME as usize;
    for i in 0..32 {
        for j in 0..8 {
            alpha[8 * i + j] = build_u64(&challenges[(primes_len * i + PRIME_BYTES_LEN * j)..]);
        }
    }

    // TODO: polxvec_jlproj_collapsmat: build constraint.linear_part from jl_matrix and hashbuf
    // TODO: jlproj_collapsproj: build constraint.constant from challenges ?
}

use crate::{
    constants::PRIME_BYTES_LEN,
    constraint::{Constraint, unpack_challenges},
    ring::BaseRingElem,
};

use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

// TODO: collapse JL ..., lift Z_q -> R_q

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
    let alpha = unpack_challenges(challenges);
    // TODO: polxvec_jlproj_collapsmat: build constraint.linear_part from jl_matrix and challenges
    //   1. pack jl_matrix as a vector of `rows: PolyVec`
    //   2. apply PolyVec::invert_x
    //   3. compute < rows[i], witness >
    //   4. compute sum, weighted by the challenges
    // TODO: jlproj_collapsproj: build constraint.constant from challenges ?
}

use crate::{
    constants::{DEGREE, PRIME_BYTES_LEN, U128_LEN},
    constraint::{Constraint, unpack_challenges},
    jl_matrix::JLMatrix,
    linear_algebra::{PolyVec, SparsePolyMatrix},
    proof::Proof,
    ring::{BaseRingElem, PolyRingElem},
    statement::Statement,
    witness::Witness,
};

use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

/// Generate a new constraint built from:
/// - the rows of each `JLMatrix`
/// - some challenges generated from `output_stat.hash`
pub fn collaps_jl_matrices(
    output_stat: &mut Statement,
    proof: &Proof,
    jl_matrices: &[JLMatrix],
) -> Constraint {
    // update hash
    let mut hashbuf = [0_u8; 32 + 256 * PRIME_BYTES_LEN];
    let mut hasher = Shake128::default();
    hasher.update(&output_stat.hash);

    let mut reader = hasher.finalize_xof();
    reader.read(&mut hashbuf);
    output_stat.hash.copy_from_slice(&hashbuf[..16]);

    // generate challenges from hashbuf[32..]
    let challenges: [BaseRingElem; 256] = unpack_challenges(&hashbuf[32..]);

    // build new constraint from challenges and JL matrices
    let mut split_linear_part: Vec<PolyVec> = vec![PolyVec::zero(output_stat.dim); output_stat.r];
    let mut constant: BaseRingElem = 0.into();

    jl_matrices
        .iter()
        .zip(split_linear_part.iter_mut())
        .for_each(|(jl_matrix, lin_part)| {
            let rows = jl_matrix.as_polyvecs_inverted();

            rows.iter().zip(challenges).for_each(|(row, challenge)| {
                *lin_part += row * &challenge;
            })
        });

    challenges
        .iter()
        .zip(proof.projection)
        .for_each(|(challenge, coord)| constant += challenge * BaseRingElem::from(coord));

    Constraint {
        degree: 1,
        quadratic_part: SparsePolyMatrix::new(),
        linear_part: PolyVec::join(&split_linear_part),
        constant: PolyRingElem::from(-constant),
    }
}

/// Compute `proof.liftin_poly` from `constraint`.
///
/// Also update `constraint.constant` and `output_stat.constraint`.
pub fn generate_lifting_poly(
    output_stat: &mut Statement,
    proof: &mut Proof,
    i: usize,
    constraint: &mut Constraint,
    witness: &Witness,
) {
    // compute result (over R_p) of linear map
    constraint.constant = -constraint
        .linear_part
        .scalar_prod(&PolyVec::join(&witness.vectors));

    proof.lifting_poly[i] = constraint.constant.clone();
    proof.lifting_poly[i].element[0] = 0.into();

    // update hash
    let mut hashbuf = [0_u8; 16 + DEGREE as usize * PRIME_BYTES_LEN];
    hashbuf[0..16].copy_from_slice(&output_stat.hash);
    hashbuf[16..].copy_from_slice(&proof.lifting_poly[i].to_le_bytes());

    let mut hasher = Shake128::default();
    hasher.update(&hashbuf);
    let mut reader = hasher.finalize_xof();
    let mut new_hashbuf = [0_u8; 64];
    reader.read(&mut new_hashbuf);

    output_stat.hash.copy_from_slice(&hashbuf[..16]);

    // generate challenge (alpha)
    let alpha: PolyRingElem = PolyRingElem::challenge_from_seed(&hashbuf[32..64]);

    if i == 0 {
        output_stat.constraint.linear_part = &constraint.linear_part * &alpha;
        output_stat.constraint.constant = &constraint.constant * &alpha;
    } else {
        output_stat.constraint.linear_part += &constraint.linear_part * &alpha;
        output_stat.constraint.constant += &constraint.constant * &alpha;
    }
}

/// Aggregate all the constant-term relations into one.
#[allow(dead_code)]
pub fn aggregate_constant_coeff(
    output_stat: &mut Statement,
    proof: &mut Proof,
    witness: &Witness,
    jl_matrices: &[JLMatrix],
) {
    for i in 0..U128_LEN {
        let mut constraint = collaps_jl_matrices(output_stat, proof, jl_matrices);
        generate_lifting_poly(output_stat, proof, i, &mut constraint, witness);
    }
}

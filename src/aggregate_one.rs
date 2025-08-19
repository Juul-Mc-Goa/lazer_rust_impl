use crate::{
    constants::{PRIME_BYTES_LEN, U128_LEN},
    constraint::{Constraint, aggregate_proj_constraints, unpack_challenges},
    jl_matrix::JLMatrix,
    proof::Proof,
    ring::{BaseRingElem, PolyRingElem},
    statement::Statement,
    witness::Witness,
};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

/// Aggregate all the constant-term relations into one.
#[allow(dead_code)]
pub fn aggregate_constant_coeff(
    output_stat: &mut Statement,
    proof: &mut Proof,
    witness: &Witness,
    jl_matrices: &[JLMatrix],
) {
    // REVIEW: hashbuf length
    // hashbuf: 32 bytes, then challenges (256 BaseRingElem), then ???
    let mut hashbuf = [0_u8; 32 + PRIME_BYTES_LEN * 256 + 24];

    let (r, dim) = (output_stat.r, output_stat.dim);

    output_stat
        .constraint
        .linear_part
        .0
        .resize(r * dim, PolyRingElem::zero());

    // `constraints` contains 256 constraints: one per output coordinate
    let constraints = Constraint::from_jl_proj(jl_matrices, &witness.vectors);
    println!("aggregate: built constraints",);

    for _ in 0..U128_LEN {
        // collaps_jlproj_raw:
        // - hashbuf <- shake128(output_stat.hash)
        // - output_stat.hash <- hashbuf[..16]
        // - use hashbuf[32..] to generate A_p challenges
        // - use challenges to generate constraint

        let mut hasher = Shake128::default();
        hasher.update(&output_stat.hash);
        let mut reader = hasher.finalize_xof();

        reader.read(&mut hashbuf);
        // update hash
        output_stat.hash.copy_from_slice(&hashbuf[..16]);

        let challenges: [BaseRingElem; 256] = unpack_challenges(&hashbuf[32..]);
        let constraint = aggregate_proj_constraints(r, dim, &constraints, &challenges);

        // copy constant to proof.lifting_poly
        proof.lifting_poly.push(constraint.constant.clone());

        hasher = Shake128::default();

        hasher.update(&output_stat.hash); // digest `hash`
        hasher.update(&constraint.constant.to_le_bytes()); // digest constant

        let mut new_hash = [0_u8; 32];
        reader.read(&mut new_hash);
        // update output_stat.hash
        output_stat.hash.copy_from_slice(&new_hash[..16]);

        // use new_hash to generate alpha
        let mut rng = ChaCha8Rng::from_seed(new_hash);
        let alpha: PolyRingElem = PolyRingElem::random(&mut rng);

        // update output_stat.constraint
        output_stat
            .constraint
            .linear_part
            .add_mul_assign(&alpha, &constraint.linear_part);
        output_stat.constraint.constant += &alpha * constraint.constant;
    }
}

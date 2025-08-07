use crate::{
    constants::{PRIME_BYTES_LEN, U128_LEN},
    constraint::{Constraint, aggregate_constraints, unpack_challenges},
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

#[allow(dead_code)]
pub fn aggregate_constant_coeff(
    output_stat: &mut Statement,
    proof: &mut Proof,
    witness: &Witness,
    dim: usize,
    jl_matrices: &[JLMatrix],
) {
    // REVIEW: hashbuf length
    // hashbuf: 32 bytes, then challenges (256 BaseRingElem), then ???
    let mut hashbuf = [0_u8; 32 + PRIME_BYTES_LEN * 256 + 24];

    // `constraints` contains 256 constraints: one per output coordinate
    let constraints = Constraint::from_jl_proj(jl_matrices, &witness.vectors, witness.r);

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
        let constraint = aggregate_constraints(dim, witness.r, &constraints, &challenges);

        // copy constant to proof.lifting_poly
        proof.lifting_poly.push(constraint.constant.clone());

        // lift_aggregate_zqcnst:
        // - new_hash <- shake128(output_stat.hash || constraint.constant)
        // - output_stat.hash <- new_hash[..16]
        // - generate alpha: PolyRingElem
        // - multiply all data in `constraint` by alpha
        // - add the result to output_stat.constraint
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
        for (stat_vec, cons_vec) in output_stat
            .constraint
            .linear_part
            .iter_mut()
            .zip(constraint.linear_part)
        {
            stat_vec.add_mul_assign(alpha.clone(), &cons_vec);
        }
        output_stat.constraint.constant += &alpha * constraint.constant;
    }
}

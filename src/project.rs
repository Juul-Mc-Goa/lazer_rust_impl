//! Implement the application of a (bitpacked) JL matrix to a vector
//! ([`add_apply_jl_matrix`]), and the [`project`] function, which generates a
//! random JL matrix and applies it to the witness.
use crate::{
    constants::JL_MAX_NORM_SQ,
    jl_matrix::JLMatrix,
    proof::Proof,
    ring::BaseRingElem,
    statement::Statement,
    utils::{Aes128Ctr64LE, next_2_power},
    witness::Witness,
};
use aes::cipher::KeyIvInit;
use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

// pub fn proj_from_bytes(bytes: &[u8; 256 * PRIME_BYTES_LEN]) -> [BaseRingElem; 256] {
//     let mut result = [BaseRingElem::zero(); 256];

//     for (i, chunk) in bytes.chunks_exact(PRIME_BYTES_LEN).enumerate() {
//         let mut bytes = [0_u8; PRIME_BYTES_LEN];
//         bytes.copy_from_slice(chunk);
//         result[i] = BaseRingElem::from_le_bytes(&bytes);
//     }

//     result
// }

/// Compute the squared norm of a vector of dimension 256 with coordinates in `A_p`.
pub fn proj_norm_square(projected: &[BaseRingElem; 256]) -> u128 {
    projected
        .iter()
        .map(|coord| {
            let abs = coord.abs() as u128;
            abs * abs
        })
        .sum::<u128>()
}

/// "Project" (not really a projection actually but anyway) `wit` into a vector
/// of dimension 256 over `A_p = Z/pZ`.  Produce the random matrix with an AES in CTR
/// mode, store the nonce in `proof.jl_nonce`.  Hash the resulting vector and
/// store it in `statement.hash`.
#[allow(dead_code)]
pub fn project(statement: &mut Statement, proof: &mut Proof, wit: &Witness) -> Vec<JLMatrix> {
    // compute total norm_square for witness
    let norm_square: u128 = wit.norm_square.iter().map(|x| *x as u128).sum();

    if norm_square > JL_MAX_NORM_SQ {
        panic!(
            "Total witness squared norm ({norm_square}) bigger than bound
                   ({JL_MAX_NORM_SQ})"
        );
    }

    let max_proj_coord = next_2_power(4 * norm_square.isqrt() as u64);
    let norm_square = norm_square * 256;

    // initialise hasher
    let mut hasher = Shake128::default();
    hasher.update(&statement.hash);
    let mut reader = hasher.finalize_xof();

    // store result in hashbuf
    let mut hashbuf = [0u8; 32];
    reader.read(&mut hashbuf);
    // split the result in two
    let (hashbuf_chunks, []) = hashbuf.as_chunks::<16>() else {
        panic!("project: hashbuf length is not a multiple of 16.");
    };
    let hashbuf_left = hashbuf_chunks[0];
    let hashbuf_right = hashbuf_chunks[1];

    proof.jl_nonce = 0;
    let (mut projection_too_big, mut coef_too_big) = (true, true);

    let mut jl_matrices: Vec<JLMatrix> = Vec::new();

    while projection_too_big || coef_too_big {
        jl_matrices = Vec::new();
        proof.projection.fill(0.into());

        let mut cipher =
            Aes128Ctr64LE::new(&hashbuf_right.into(), &proof.jl_nonce.to_le_bytes().into());

        for i in 0..wit.r {
            let jl_matrix = JLMatrix::random(wit.dim[i], &mut cipher);

            jl_matrix.add_apply(&mut proof.projection, &wit.vectors[i]);
            jl_matrices.push(jl_matrix);
        }

        // increment nonce
        proof.jl_nonce += 1;

        // update booleans
        coef_too_big = proof
            .projection
            .iter()
            .any(|coord| coord.abs() >= max_proj_coord);
        projection_too_big = proj_norm_square(&proof.projection) > norm_square;
    }

    // reverse last increment
    proof.jl_nonce -= 1;

    // initialise hasher: send hashbuf[..16] + proof.projection as input
    let mut hasher = Shake128::default();
    // digest the left part of `hashbuf` and the bytes in `projection`
    hasher.update(&hashbuf_left);
    hasher.update(
        &proof
            .projection
            .iter()
            .map(|e| e.to_le_bytes())
            .flatten()
            .collect::<Vec<_>>(),
    );

    let mut reader = hasher.finalize_xof();

    // store hash in statement.hash
    reader.read(&mut statement.hash);

    jl_matrices
}

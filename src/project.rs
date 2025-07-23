//! Implement the application of a (bitpacked) JL matrix to a vector
//! ([`add_apply_jl_matrix`]), and the [`project`] function, which generates a
//! random JL matrix and applies it to the witness.
use crate::{
    constants::{DEGREE, JL_MAX_NORM_SQ},
    linear_algebra::PolyVec,
    proof::Proof,
    ring::BaseRingElem,
    statement::Statement,
    utils::{Aes128Ctr64LE, next_2_power},
    witness::Witness,
};
use aes::cipher::{KeyIvInit, StreamCipher};
use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

/// Apply `jl_matrix` to `in_vec`, add the result to `out_vec`.
#[allow(dead_code)]
pub fn add_apply_jl_matrix(
    out_vec: &mut [BaseRingElem; 256],
    in_vec: &PolyVec,
    jl_matrix: &Vec<u8>,
) {
    let mut jl_bit_idx = 0;

    for v in &in_vec.0 {
        for coord in 0..256 {
            for coef in &v.element {
                let byte_idx = jl_bit_idx >> 3;
                let mask = 1 << (jl_bit_idx & 7);

                if (jl_matrix[byte_idx] & mask) != 0 {
                    out_vec[coord] -= coef;
                } else {
                    out_vec[coord] += coef;
                }

                jl_bit_idx += 1;
            }
        }
    }
}

/// "Project" (not really a projection actually but anyway) `wit` into a vector
/// of dimension 256 over `Z/pZ`.  Produce the random matrix with an AES in CTR
/// mode, store the nonce in `proof.jl_nonce`.  Hash the resulting vector and
/// store it in `statement.hash`.
#[allow(dead_code)]
pub fn project(statement: &mut Statement, proof: &mut Proof, wit: Witness) {
    // compute total norm_square for witness
    let norm_square: u128 = wit.norm_square.iter().map(|x| *x as u128).sum();

    if norm_square > JL_MAX_NORM_SQ {
        eprintln!(
            "Total witness squared norm ({norm_square}) bigger than bound
                   ({JL_MAX_NORM_SQ})"
        );
        return;
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
        panic!("hashbuf length is not a multiple of 16.");
    };
    let hashbuf_left = hashbuf_chunks[0];
    let hashbuf_right = hashbuf_chunks[1];

    proof.jl_nonce = 0;
    let (mut projection_too_big, mut coef_too_big) = (true, true);

    while projection_too_big || coef_too_big {
        let mut jl_matrices: Vec<Vec<u8>> = Vec::new();
        let mut cipher =
            Aes128Ctr64LE::new(&hashbuf_right.into(), &proof.jl_nonce.to_le_bytes().into());
        proof.projection.fill(BaseRingElem::zero());

        for i in 0..wit.r {
            // 1 bit per (in, out) coordinate
            // input: wit.dim[i] * DEGREE coordinates
            // output: 256 coordinates
            let jl_bytes = wit.dim[i] * DEGREE as usize * 256 / 8;
            let mut jl_matrix = vec![0_u8; jl_bytes];

            // generate jl_matrix
            cipher.apply_keystream(&mut jl_matrix);

            // apply the matrix to witness.vectors
            add_apply_jl_matrix(&mut proof.projection, &wit.vectors[i], &jl_matrix);
            jl_matrices.push(jl_matrix);
        }

        // increment nonce
        proof.jl_nonce += 1;

        // update booleans
        coef_too_big = proof
            .projection
            .iter()
            .any(|coord| coord.element >= max_proj_coord);
        projection_too_big = proof
            .projection
            .iter()
            .map(|coord| coord.element as u128 * coord.element as u128)
            .sum::<u128>()
            > norm_square;
    }

    // initialise hasher: send hashbuf[..16] + proof.projection as input
    let mut hasher = Shake128::default();
    // digest the left part of `hashbuf` and the bytes in `projection`
    hasher.update(&hashbuf_left);
    hasher.update(
        &proof
            .projection
            .iter()
            .map(|e| e.element.to_le_bytes())
            .collect::<Vec<_>>()
            .as_flattened(),
    );

    let mut reader = hasher.finalize_xof();

    // store hash in statement.hash
    reader.read(&mut statement.hash);
}

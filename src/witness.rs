use crate::{
    constants::{DEGREE, JL_MAX_NORM_SQ},
    proof::Proof,
    ring::{BaseRingElem, PolyRingElem},
    statement::Statement,
};
use aes::cipher::{KeyIvInit, StreamCipher};
use ctr;
use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

pub type Aes128Ctr64LE = ctr::Ctr64LE<aes::Aes128>;

#[allow(dead_code)]
pub struct Witness {
    /// Number of vectors (ie length of `self.vectors`).
    pub r: usize,
    /// Dimension of each vector.
    pub dim: Vec<usize>,
    /// Squared norm of each vector
    pub norm_square: Vec<u128>,
    /// The list of vectors.
    pub vectors: Vec<Vec<PolyRingElem>>,
}

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

/// Apply `jl_matrix` to `in_vec`, add the result to `out_vec`.
#[allow(dead_code)]
pub fn add_apply_jl_matrix(
    out_vec: &mut [BaseRingElem; 256],
    in_vec: &Vec<PolyRingElem>,
    jl_matrix: &Vec<u8>,
) {
    let mut jl_bit_idx = 0;

    // assert_eq!(jl_matrix[0] & 1, 1);
    println!("input dim: {}", in_vec.len());
    println!("jl dim: {}", jl_matrix.len());

    for v in in_vec {
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

impl Witness {
    /// Create a witness with correct capacities.
    pub fn new_raw(r: usize, dim: Vec<usize>) -> Self {
        let len = dim.iter().sum();
        Self {
            r,
            dim,
            norm_square: Vec::with_capacity(r),
            vectors: Vec::with_capacity(len),
        }
    }

    pub fn new(statement: &Statement) -> Self {
        let mut r = statement.r;
        let mut dim = vec![statement.dim; statement.commit_params.z_length];

        if !statement.tail {
            r += 1;
            dim.push(statement.dim_inner);
        }

        Self::new_raw(r, dim)
    }
}

/// Project `wit` into a vector of dimension 256 over `Z/pZ`.  Produce the
/// random matrix with an AES in CTR mode, store the nonce in `proof.jl_nonce`.
/// Hash the resulting vector and store it in `statment.hash`.
#[allow(dead_code)]
pub fn wit_project(statement: &mut Statement, proof: &mut Proof, wit: Witness) {
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
    let mut jl_matrices: Vec<Vec<u8>> = Vec::new();
    let (mut projection_too_big, mut coef_too_big) = (true, true);

    while projection_too_big || coef_too_big {
        let mut cipher =
            Aes128Ctr64LE::new(&hashbuf_right.into(), &proof.jl_nonce.to_le_bytes().into());
        proof.projection.fill(BaseRingElem::zero());

        for i in 0..wit.r {
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
    // concatenate the left part of `hashbuf` and the bytes in `projection`
    let hash_input: Vec<u8> = hashbuf_left
        .iter()
        .chain(
            proof
                .projection
                .iter()
                .map(|e| e.element.to_be_bytes())
                .collect::<Vec<_>>()
                .as_flattened()
                .iter(),
        )
        .map(|u_ref| *u_ref)
        .collect();
    hasher.update(&hash_input);

    let mut reader = hasher.finalize_xof();

    // store hash in statement.hash
    reader.read(&mut statement.hash);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_jl_mat() {
        let mut projection = [BaseRingElem::from(0); 256];
        let in_dim = 4;

        let mut in_vec = vec![PolyRingElem::zero(); in_dim];
        in_vec[0] = PolyRingElem::one();
        in_vec[1] = PolyRingElem::one();

        let jl_bytes = in_dim * DEGREE as usize * 256 / 8;
        let jl_matrix = vec![0_u8; jl_bytes];

        add_apply_jl_matrix(&mut projection, &in_vec, &jl_matrix);

        assert_eq!(projection, [2.into(); 256])
    }

    #[test]
    fn jl_mat_sub_coef() {
        let mut projection = [BaseRingElem::from(0); 256];

        let in_dim = 4;
        let mut in_vec = vec![PolyRingElem::zero(); in_dim];

        let mut first_poly = vec![0_u64; DEGREE as usize];
        first_poly[0] = 1;
        first_poly[1] = 1;
        in_vec[0] = PolyRingElem::from_vec_u64(first_poly);

        let jl_bytes = in_dim * DEGREE as usize * 256 / 8;
        let mut jl_matrix = vec![0_u8; jl_bytes];
        // each 8-byte (64-bit) block is dedicated to one coordinate of
        // projection and one coordinate of in_vec
        jl_matrix[0] = 3;

        add_apply_jl_matrix(&mut projection, &in_vec, &jl_matrix);

        assert_eq!(projection[0], -BaseRingElem::from(2));
        for c in projection[1..].iter() {
            assert_eq!(c, &BaseRingElem::from(2))
        }
    }

    #[test]
    fn jl_mat_out_coord() {
        let mut projection = [BaseRingElem::from(0); 256];

        let in_dim = 4;
        let mut in_vec = vec![PolyRingElem::zero(); in_dim];
        in_vec[0] = PolyRingElem::one();
        in_vec[1] = PolyRingElem::one();

        let jl_bytes = in_dim * DEGREE as usize * 256 / 8;
        let mut jl_matrix = vec![0_u8; jl_bytes];
        // each 8-byte (64-bit) block is dedicated to one coordinate of
        // projection and one coordinate of in_vec
        jl_matrix[0] = 1;
        jl_matrix[8] = 1;

        add_apply_jl_matrix(&mut projection, &in_vec, &jl_matrix);

        assert_eq!(projection[0], BaseRingElem::from(0));
        assert_eq!(projection[1], BaseRingElem::from(0));
        for c in projection[2..].iter() {
            assert_eq!(c, &BaseRingElem::from(2))
        }
    }

    #[test]
    fn jl_mat_sub_vectors() {
        let mut projection = [BaseRingElem::from(0); 256];
        let in_dim = 4;
        let mut in_vec = vec![PolyRingElem::zero(); in_dim];
        in_vec[0] = PolyRingElem::one();
        in_vec[1] = PolyRingElem::one();

        let jl_bytes = in_dim * DEGREE as usize * 256 / 8;
        let mut jl_matrix = vec![0_u8; jl_bytes];
        // each 2048-byte (= 16348 bits = 256 * 64 bits) block is dedicated to
        // one coordinate in in_vec
        jl_matrix[0] = 1;
        jl_matrix[2048] = 1;

        add_apply_jl_matrix(&mut projection, &in_vec, &jl_matrix);

        assert_eq!(projection[0], -BaseRingElem::from(2));
        for c in projection[1..].iter() {
            assert_eq!(c, &BaseRingElem::from(2))
        }
    }
}

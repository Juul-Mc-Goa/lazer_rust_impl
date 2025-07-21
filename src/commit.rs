use crate::{
    constants::{DEGREE, LOG_PRIME},
    proof::Proof,
    ring::{PolyRingElem, poly_vec_decomp},
    statement::Statement,
    witness::Witness,
};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[allow(dead_code)]
#[derive(Clone)]
pub struct CommitParams {
    pub z_base: usize,
    pub z_length: usize,
    pub uniform_base: usize,
    pub uniform_length: usize,
    pub quadratic_base: usize,
    pub quadratic_length: usize,
    pub commit_rank_1: usize,
    pub commit_rank_2: usize,
    pub u1_len: usize,
    pub u2_len: usize,
}

pub struct CommitKey {
    data: Vec<PolyRingElem>,
    // inner_data
    //
    seed: <ChaCha8Rng as SeedableRng>::Seed,
}

pub enum Commitments {
    Tail {
        inner: Vec<PolyRingElem>,
        garbage: Vec<PolyRingElem>,
    },
    NoTail {
        u1: Vec<PolyRingElem>,
        u2: Vec<PolyRingElem>,
    },
}

impl CommitKey {
    pub fn new(len: usize) -> Self {
        // generate seed
        let mut seed: <ChaCha8Rng as SeedableRng>::Seed = Default::default();
        rand::rng().fill(&mut seed);

        //generate data
        let mut rng = ChaCha8Rng::from_seed(seed);
        let data: Vec<PolyRingElem> = (0..len).map(|_| PolyRingElem::random(&mut rng)).collect();

        Self { data, seed }
    }

    pub fn apply_matrix(&self, in_vec: &Vec<PolyRingElem>, out_dim: usize) -> Vec<PolyRingElem> {
        let mut out_vec = vec![PolyRingElem::zero(); out_dim];

        let mut i = 0;
        for line in self.data.chunks(in_vec.len()).take(out_dim) {
            for (l_coef, v_coord) in line.iter().zip(in_vec.iter()) {
                out_vec[i] += l_coef * v_coord;
            }
            i += 1;
        }

        if i + 1 < out_dim {
            panic!(
                "Commitment key is too small: {} < {} * {}",
                self.data.len(),
                in_vec.len(),
                out_vec.len(),
            );
        }

        out_vec
    }
}

impl Commitments {
    pub fn new(tail: bool) -> Self {
        if tail {
            Self::Tail {
                inner: Vec::new(),
                garbage: Vec::new(),
            }
        } else {
            Self::NoTail {
                u1: Vec::new(),
                u2: Vec::new(),
            }
        }
    }
}

fn commit_tail(
    witness: &[Vec<PolyRingElem>],
    commit_key: &CommitKey,
    com_params: &CommitParams,
) -> Vec<PolyRingElem> {
    let l = com_params.uniform_length * com_params.commit_rank_1;
    let mut result: Vec<PolyRingElem> = Vec::with_capacity(witness.len() * l);

    for i in 0..witness.len() {
        result.append(&mut commit_key.apply_matrix(&witness[i], com_params.commit_rank_1));
    }

    result
}

fn commit_no_tail(
    u1: &mut Vec<PolyRingElem>,
    witness: &[Vec<PolyRingElem>],
    commit_key: &CommitKey,
    com_params: &CommitParams,
) -> Vec<Vec<PolyRingElem>> {
    // TODO: - apply offset matrix to get outer commit summand

    let (com_rank1, unif_base, unif_len) = (
        com_params.commit_rank_1,
        com_params.uniform_base,
        com_params.uniform_length,
    );
    let mut inner_commit: Vec<PolyRingElem> = Vec::with_capacity(com_rank1 * unif_len);

    let mut inner_decomp: Vec<Vec<PolyRingElem>> = Vec::new();

    for i in 0..witness.len() {
        inner_commit.append(&mut commit_key.apply_matrix(&witness[i], com_rank1));

        // decompose inner_commit
        let com_start_idx = inner_commit.len() - com_rank1;
        inner_decomp.append(&mut poly_vec_decomp(
            &inner_commit[com_start_idx..],
            unif_base,
        ));
    }

    inner_decomp
}

pub fn commit(
    commit_key: &CommitKey,
    output_stat: &mut Statement,
    output_wit: &mut Witness,
    proof: &mut Proof,
    input_wit: &Witness,
) -> Vec<Vec<PolyRingElem>> {
    let r = input_wit.r;
    let dim = &input_wit.dim;
    let com_params = &output_stat.commit_params;

    let hashbuf_len = 16 + com_params.u1_len * DEGREE as usize * (LOG_PRIME >> 3) as usize;
    let mut hashbuf = vec![0_u8; hashbuf_len];

    // out_vec is the concatenation of (commitment to iwt.vectors) and (garbage)
    let out_vec = &mut output_wit.vectors;

    let mut full_witness: Vec<Vec<PolyRingElem>> =
        Vec::with_capacity(output_stat.r * output_stat.dim);

    let mut vector_idx = 0; // in 0..r
    let mut coord_idx = 0; // in 0..n

    for i in 0..r {
        full_witness[vector_idx].extend_from_slice(&input_wit.vectors[i]);
        coord_idx += dim[i];

        if proof.wit_length[i] != 0 {
            let coords_to_fill = output_stat.dim - coord_idx;
            let vectors_to_fill = proof.wit_length[i] - 1;

            // fill the rest of commits[vector_idx] with zeros
            for _ in 0..coords_to_fill {
                full_witness[vector_idx].push(PolyRingElem::zero());
            }

            // each vector in the decomposition of input_wit.vectors[i] is
            // filled with zeros
            for _ in 0..vectors_to_fill {
                full_witness.push(vec![PolyRingElem::zero(); output_stat.dim]);
            }

            if proof.tail {
                output_stat.commitments.u1.append(&mut commit_tail(
                    &full_witness,
                    commit_key,
                    com_params,
                ))
            } else {
                out_vec.append(&mut commit_no_tail(&full_witness, commit_key, com_params))
            }

            // update vector index: filled exactly wit_length[i] vectors
            vector_idx += proof.wit_length[i];
            // next vector: start at the first coordinate
            coord_idx = 0;
        }
    }
    // TODO: quadratic garbage
    // TODO: copy u1 from output_stat to proof
    // TODO: bitpack output_stat.hash, shake128

    // FIXME
    Vec::new()
}

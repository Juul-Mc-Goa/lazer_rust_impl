use crate::{
    linear_algebra::{PolyMatrix, PolyVec},
    proof::Proof,
    ring::PolyRingElem,
    statement::Statement,
    witness::Witness,
};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

#[allow(dead_code)]
#[derive(Copy, Clone)]
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

#[allow(dead_code)]
pub enum CommitKeyData {
    /// The matrices `A, B, C` used to compute
    /// `u1 = B * decomp(A * z) + C * decomp(quad_garbage)`.
    Tail {
        /// matrix of size `commit_rank_1 * dim`
        matrix_a: PolyMatrix,
        /// `r * z_length` matrices of size `commit_rank_2 * commit_rank_1`
        matrices_b: Vec<Vec<PolyMatrix>>,
        /// `r * quad_length` matrices, where `matrix_c[i][k]` has size `commit_rank_2 * i`.
        matrices_c: Vec<Vec<PolyMatrix>>,
    },
    /// The matrices `A, B, C, D` used to compute:
    /// - `u1 = B * decomp(A * z) + C * decomp(quad_garbage)`
    /// - `u2 = D * decomp(unif_garbage)`
    NoTail {
        /// matrix of size `commit_rank_1 * dim`
        matrix_a: PolyMatrix,
        /// `r * z_length` matrices of size `commit_rank_2 * commit_rank_1`
        matrices_b: Vec<Vec<PolyMatrix>>,
        /// `r * quad_length` matrices, where `matrix_c[i][k]` has size `commit_rank_2 * i`.
        matrices_c: Vec<Vec<PolyMatrix>>,
        /// `r * uniform_length` matrices, where `matrix_d[i][k]` has size `commit_rank_2 * i`.
        matrices_d: Vec<Vec<PolyMatrix>>,
    },
}

#[allow(dead_code)]
pub struct CommitKey {
    /// The matrices used for commiting.
    data: CommitKeyData,
    /// Seed used to generate `self.data`
    seed: <ChaCha8Rng as SeedableRng>::Seed,
}

#[allow(dead_code)]
#[derive(Clone)]
pub enum Commitments {
    Tail {
        inner: PolyVec,
        garbage: PolyVec,
    },
    NoTail {
        inner: PolyVec,
        u1: PolyVec,
        u2: PolyVec,
    },
}

#[allow(dead_code)]
impl CommitKey {
    pub fn new(tail: bool, r: usize, dim: usize, com_params: CommitParams) -> Self {
        // generate seed
        let mut seed: <ChaCha8Rng as SeedableRng>::Seed = Default::default();
        rand::rng().fill(&mut seed);

        // generate data
        let mut rng = ChaCha8Rng::from_seed(seed);

        let matrix_a: PolyMatrix = PolyMatrix::random(&mut rng, com_params.commit_rank_1, dim);
        let matrices_b: Vec<Vec<PolyMatrix>> = (0..r)
            .map(|_| {
                (0..com_params.z_length)
                    .map(|_| {
                        PolyMatrix::random(
                            &mut rng,
                            com_params.commit_rank_2,
                            com_params.commit_rank_1,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let matrices_c: Vec<Vec<PolyMatrix>> = (0..r)
            .map(|i| {
                (0..com_params.quadratic_length)
                    .map(|_| PolyMatrix::random(&mut rng, com_params.commit_rank_2, i))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        if tail {
            Self {
                data: CommitKeyData::Tail {
                    matrix_a,
                    matrices_b,
                    matrices_c,
                },
                seed,
            }
        } else {
            let matrices_d: Vec<Vec<PolyMatrix>> = (0..r)
                .map(|i| {
                    (0..com_params.uniform_length)
                        .map(|_| PolyMatrix::random(&mut rng, com_params.commit_rank_2, i))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            Self {
                data: CommitKeyData::NoTail {
                    matrix_a,
                    matrices_b,
                    matrices_c,
                    matrices_d,
                },
                seed,
            }
        }
    }
}

impl Commitments {
    pub fn new(tail: bool) -> Self {
        if tail {
            Self::Tail {
                inner: PolyVec::new(),
                garbage: PolyVec::new(),
            }
        } else {
            Self::NoTail {
                inner: PolyVec::new(),
                u1: PolyVec::new(),
                u2: PolyVec::new(),
            }
        }
    }
}

#[allow(dead_code)]
fn commit_tail(inner: &mut PolyVec, witness: &[PolyVec], matrix_a: &PolyMatrix) {
    // apply the matrix A to each vector in the witness
    for w in witness {
        inner.0.append(&mut matrix_a.apply(w).0);
    }
}

#[allow(dead_code)]
fn commit_no_tail(
    outer: &mut PolyVec,
    inner: &mut PolyVec,
    witness: &[PolyVec],
    matrix_a: &PolyMatrix,
    matrices_b: &Vec<Vec<PolyMatrix>>,
    com_params: &CommitParams,
) {
    let unif_base = com_params.uniform_base;

    for i in 0..witness.len() {
        let mut chunk = matrix_a.apply(&witness[i]);

        // decompose inner_commit, apply the matrices in matrices_b
        for (k, inner_decomp) in chunk.decomp(unif_base).iter().enumerate() {
            matrices_b[i][k].add_apply(outer, inner_decomp);
        }

        inner.0.append(&mut chunk.0);
    }
}

#[allow(dead_code)]
pub fn commit(
    commit_key: &CommitKey,
    output_stat: &mut Statement,
    output_wit: &mut Witness,
    proof: &mut Proof,
    input_wit: &Witness,
) {
    let (r, dim) = (input_wit.r, &input_wit.dim);
    let com_params = &output_stat.commit_params;

    // full_witness is the concatenation of (commitment to iwt.vectors) and (garbage)
    let mut full_witness: Vec<PolyVec> = Vec::with_capacity(output_stat.r * output_stat.dim);

    let mut vector_idx = 0; // in 0..r
    let mut coord_idx = 0; // in 0..n

    output_stat.commitments = Commitments::new(proof.tail);

    // first step: handle inner commitment (using input_wit)
    for i in 0..r {
        full_witness[vector_idx]
            .0
            .extend_from_slice(&input_wit.vectors[i].0);
        coord_idx += dim[i];

        if proof.wit_length[i] != 0 {
            let coords_to_fill = output_stat.dim - coord_idx;
            let vectors_to_fill = proof.wit_length[i] - 1;

            // fill the rest of full_witness[vector_idx] with zeros
            for _ in 0..coords_to_fill {
                full_witness[vector_idx].0.push(PolyRingElem::zero());
            }

            // each vector (except the first) in the decomposition of
            // input_wit.vectors[i] is filled with zeros
            for _ in 0..vectors_to_fill {
                full_witness.push(PolyVec::zero(output_stat.dim));
            }

            match (&mut output_stat.commitments, &commit_key.data) {
                (
                    &mut Commitments::Tail { ref mut inner, .. },
                    &CommitKeyData::Tail { ref matrix_a, .. },
                ) => commit_tail(inner, &full_witness[vector_idx..], matrix_a),
                (
                    &mut Commitments::NoTail {
                        ref mut inner,
                        ref mut u1,
                        ..
                    },
                    &CommitKeyData::NoTail {
                        ref matrix_a,
                        ref matrices_b,
                        ..
                    },
                ) => commit_no_tail(
                    u1,
                    inner,
                    &mut full_witness[vector_idx..],
                    matrix_a,
                    matrices_b,
                    com_params,
                ),
                _ => panic!(
                    "output_stat.commitments and commit_key.data have incompatible `Tail` variants"
                ),
            }

            // update vector index: filled exactly wit_length[i] vectors
            vector_idx += proof.wit_length[i];
            // next vector: start at the first coordinate
            coord_idx = 0;
        }
    }

    // second step: handle garbage
    if com_params.quadratic_length != 0 {
        // compute quadratic garbage < s_i, s_j >
        let mut quad_inner: Vec<Vec<PolyRingElem>> = Vec::new();
        for i in 0..output_stat.r {
            quad_inner.push(vec![PolyRingElem::zero(); output_stat.r]);
            for j in 0..=i {
                for k in 0..output_stat.dim {
                    quad_inner[i][j] += &full_witness[i].0[k] * &full_witness[j].0[k];
                }
            }
        }

        match output_stat.commitments {
            Commitments::Tail {
                inner: _,
                ref mut garbage,
            } => {
                // tail: append the quadratic garbage to commitments.garbage
                for mut garbage_vec in quad_inner.drain(..) {
                    garbage.0.append(&mut garbage_vec);
                }
            }
            Commitments::NoTail {
                inner: _,
                ref mut u1,
                u2: _,
            } => {
                // no tail:
                // - decompose quad_inner,
                // - apply matrices_c, update u1
                // - append quad_inner to output_wit.vectors

                let CommitKeyData::NoTail {
                    matrix_a: _,
                    matrices_b: _,
                    ref matrices_c,
                    matrices_d: _,
                } = commit_key.data
                else {
                    panic!("commit key data is `Tail` but commitments are `NoTail`");
                };

                // decomp_inner[i][k][j]: level k of < s_i, s_j >
                let decomp_inner: Vec<Vec<PolyVec>> = quad_inner
                    .into_iter()
                    .map(|line| PolyVec(line).decomp(com_params.quadratic_base))
                    .collect();

                for (levels_matrices, mut levels_inner) in
                    matrices_c.iter().zip(decomp_inner.into_iter())
                {
                    for (level_matrix, level_inner) in
                        levels_matrices.iter().zip(levels_inner.iter())
                    {
                        // update u1
                        level_matrix.add_apply(u1, &level_inner);
                    }

                    output_wit.vectors.append(&mut levels_inner);
                }
            }
        }
    }

    // copy commitments from output_stat to proof
    proof.commitments = output_stat.commitments.clone();

    let mut hasher = Shake128::default();
    hasher.update(&output_stat.hash);

    match output_stat.commitments {
        Commitments::Tail {
            ref inner,
            ref garbage,
        } => {
            // hash inner and garbage
            for byte in inner.iter_bytes().chain(garbage.iter_bytes()) {
                hasher.update(&[byte]);
            }
        }
        Commitments::NoTail {
            inner: _,
            ref u1,
            u2: _,
        } => {
            // hash u1
            for byte in u1.iter_bytes() {
                hasher.update(&[byte]);
            }
        }
    }

    let mut reader = hasher.finalize_xof();

    // store hash in output_hash
    reader.read(&mut output_stat.hash);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_tail_simple_a() {
        let dim = 3;
        let r = 4;
        let com_rank_1 = 3;
        // create a bidiagonal matrix
        let matrix_a: PolyMatrix = PolyMatrix(
            (0..com_rank_1)
                .map(|i| {
                    let mut line = vec![PolyRingElem::zero(); dim];
                    line[i] = PolyRingElem::one();
                    line[(i + 1) % dim] = PolyRingElem::one();
                    line
                })
                .collect(),
        );

        // generate random witness
        let mut seed: <ChaCha8Rng as SeedableRng>::Seed = Default::default();
        rand::rng().fill(&mut seed);
        let mut rng = ChaCha8Rng::from_seed(seed);
        let witness: Vec<PolyVec> = (0..r).map(|_| PolyVec::random(dim, &mut rng)).collect();

        let mut expected_result: Vec<PolyVec> = vec![PolyVec::zero(com_rank_1); dim];

        for i in 0..dim {
            for j in 0..com_rank_1 {
                expected_result[i].0[j] = &witness[i].0[j] + &witness[i].0[(j + 1) % dim];
            }
        }

        let mut inner = PolyVec::new();
        commit_tail(&mut inner, &witness, &matrix_a);

        for (chunk, res) in inner.0.chunks_exact(com_rank_1).zip(expected_result.iter()) {
            assert_eq!(chunk, res.0)
        }
    }
}

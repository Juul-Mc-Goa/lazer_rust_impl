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
#[derive(Copy, Clone, Debug)]
pub struct CommitParams {
    /// The base in which to decompose `z = sum(i, c_i s_i)`.
    pub z_base: usize,
    /// The number of limbs `z` have in `z_base`.
    pub z_length: usize,
    /// The base in which to decompose each `t_i = As_i` and each `h_ij`.
    pub uniform_base: usize,
    /// The max number of limbs `t_i, h_ij` have in `uniform_base`.
    pub uniform_length: usize,
    /// The base in which to decompose each `g_ij`.
    pub quadratic_base: usize,
    /// The max number of limbs `g_ij` have in `quadratic_base`.
    pub quadratic_length: usize,
    /// The rank of each `t_i` (the inner commitments).
    pub commit_rank_1: usize,
    /// The rank of `u1, u2` (the outer commitments).
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
        /// `z_length * r` matrices of size `commit_rank_2 * commit_rank_1`
        matrices_b: Vec<Vec<PolyMatrix>>,
        /// `quad_length * r` matrices, where `matrix_c[i][k]` has size `commit_rank_2 * (r - i)`.
        matrices_c: Vec<Vec<PolyMatrix>>,
    },
    /// The matrices `A, B, C, D` used to compute:
    /// - `u1 = B * decomp(A * z) + C * decomp(quad_garbage)`
    /// - `u2 = D * decomp(unif_garbage)`
    NoTail {
        /// matrix of size `commit_rank_1 * dim`
        matrix_a: PolyMatrix,
        /// `z_length * r` matrices of size `commit_rank_2 * commit_rank_1`
        matrices_b: Vec<Vec<PolyMatrix>>,
        /// `quad_length * r` matrices, where `matrix_c[k][i]` has size `commit_rank_2 * (r - i)`.
        matrices_c: Vec<Vec<PolyMatrix>>,
        /// `uniform_length * r` matrices, where `matrix_d[k][i]` has size `commit_rank_2 * (r - i)`.
        matrices_d: Vec<Vec<PolyMatrix>>,
    },
}

#[allow(dead_code)]
pub struct CommitKey {
    /// The matrices used for commiting.
    pub data: CommitKeyData,
    /// Seed used to generate `self.data`.
    pub seed: <ChaCha8Rng as SeedableRng>::Seed,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
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
        let matrices_b: Vec<Vec<PolyMatrix>> = (0..com_params.z_length)
            .map(|_| {
                (0..r)
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
        let matrices_c: Vec<Vec<PolyMatrix>> = (0..com_params.quadratic_length)
            .map(|i| {
                (0..r)
                    .map(|_| PolyMatrix::random(&mut rng, com_params.commit_rank_2, r - i))
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
            let matrices_d: Vec<Vec<PolyMatrix>> = (0..com_params.uniform_length)
                .map(|i| {
                    (0..r)
                        .map(|_| PolyMatrix::random(&mut rng, com_params.commit_rank_2, r - i))
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
fn inner_commit_tail(inner: &mut PolyVec, witness: &[PolyVec], matrix_a: &PolyMatrix) {
    // apply the matrix A to each vector in the witness
    for w in witness {
        inner.0.append(&mut matrix_a.apply(w).0);
    }
}

#[allow(dead_code)]
fn inner_commit_no_tail(
    outer: &mut PolyVec,
    inner: &mut PolyVec,
    outer_wit: &mut Vec<PolyVec>,
    input_wit: &[PolyVec],
    matrix_a: &PolyMatrix,
    matrices_b: &Vec<Vec<PolyMatrix>>,
    com_params: &CommitParams,
) {
    let unif_base = com_params.uniform_base;
    let unif_len = com_params.uniform_length;
    outer_wit.push(PolyVec::new());
    let last_vec = &mut outer_wit[0];

    for i in 0..input_wit.len() {
        let mut chunk = matrix_a.apply(&input_wit[i]);

        // decompose inner_commit
        for (k, inner_decomp) in chunk.decomp(unif_base, unif_len).iter().enumerate() {
            // apply the matrices in matrices_b
            matrices_b[k][i].add_apply(outer, inner_decomp);
            // extend last_vec
            last_vec.concat(&mut inner_decomp.clone());
        }

        inner.0.append(&mut chunk.0);
    }
}

/// Commit:
/// - modify `output_stat.commitments`
///   - if `proof.tail` is true:
///     - inner commitment is all `t_i = As_i`
///     - garbage is all `< s_i, s_j >`
///   - if `proof.tail` is false:
///     - inner commitment is all `t_i = As_i`
///     - `u1 = Bt + Cg`
/// - modify `output_wit`
///   - resize `input_wit.s_i` and append to `output_wit`
///   - push a last element: `t || g || h`
#[allow(dead_code)]
pub fn commit(
    commit_key: &CommitKey,
    output_stat: &mut Statement,
    output_wit: &mut Witness,
    proof: &mut Proof,
    input_wit: &Witness,
) {
    let r = input_wit.r;
    let com_params = &output_stat.commit_params;

    let mut full_witness = vec![PolyVec(Vec::with_capacity(output_stat.dim)); output_stat.r];

    let mut vector_idx = 0; // in 0..r

    output_stat.commitments = Commitments::new(proof.tail);
    let mut inner = PolyVec::new();
    let mut garbage = PolyVec::new();
    let mut u1 = PolyVec::new();
    let u2 = PolyVec::new();

    let matrix_a: &PolyMatrix;
    let matrices_b: &Vec<Vec<PolyMatrix>>;
    let matrices_c: &Vec<Vec<PolyMatrix>>;

    match &commit_key.data {
        CommitKeyData::Tail {
            matrix_a: a,
            matrices_b: b,
            matrices_c: c,
        } => {
            matrix_a = a;
            matrices_b = b;
            matrices_c = c;
        }
        CommitKeyData::NoTail {
            matrix_a: a,
            matrices_b: b,
            matrices_c: c,
            matrices_d: _,
        } => {
            matrix_a = a;
            matrices_b = b;
            matrices_c = c;
        }
    }

    // first step: handle inner commitment
    for i in 0..r {
        full_witness[vector_idx]
            .0
            .extend_from_slice(&input_wit.vectors[i].0);

        if proof.chunks[i] != 0 {
            // fill the rest of full_witness[vector_idx] with zeros
            full_witness[vector_idx]
                .0
                .resize(output_stat.dim, PolyRingElem::zero());

            // each vector (except the first) in the decomposition of
            // input_wit.vectors[i] is filled with zeros
            for _ in 0..(proof.chunks[i] - 1) {
                full_witness.push(PolyVec::zero(output_stat.dim));
            }

            let witness_chunk = &full_witness[vector_idx..(vector_idx + proof.chunks[i])];

            if proof.tail {
                inner_commit_tail(&mut inner, witness_chunk, matrix_a);
            } else {
                inner_commit_no_tail(
                    &mut u1,
                    &mut inner,
                    &mut output_wit.vectors,
                    witness_chunk,
                    matrix_a,
                    matrices_b,
                    com_params,
                );
            }

            // update vector index: filled exactly chunks[i] vectors
            vector_idx += proof.chunks[i];
        }
    }

    // second step: handle garbage
    if com_params.quadratic_length != 0 {
        // compute quadratic garbage < s_i, s_j >
        let mut quad_inner: Vec<Vec<PolyRingElem>> = Vec::new();
        for i in 0..output_stat.r {
            quad_inner.push(vec![PolyRingElem::zero(); i + 1]);
            for j in 0..=i {
                for k in 0..output_stat.dim {
                    quad_inner[i][j] += &full_witness[i].0[k] * &full_witness[j].0[k];
                }
            }
        }

        if proof.tail {
            // append the quadratic garbage to commitments.garbage
            for mut garbage_vec in quad_inner.drain(..) {
                garbage.0.append(&mut garbage_vec);
            }
        } else {
            // - decompose quad_inner,
            // - apply matrices_c, update u1
            // - append decomposed quad_inner to output_wit.vectors
            let quad_base = com_params.quadratic_base;
            let quad_len = com_params.quadratic_length;

            // decomp_inner[i][k][j]: level k of < s_i, s_j >
            let decomp_inner: Vec<Vec<PolyVec>> = quad_inner
                .into_iter()
                .map(|line| PolyVec(line).decomp(quad_base, quad_len))
                .collect();

            // reindex decomp_inner
            let mut g_witness: Vec<PolyVec> = vec![PolyVec::new(); quad_len];
            for i in 0..output_stat.r {
                // update u1
                for k in 0..quad_len {
                    matrices_c[k][i].add_apply(&mut u1, &decomp_inner[i][k]);
                    g_witness[k].0.append(&mut decomp_inner[i][k].0.clone());
                }
            }
            // ie append one vector containing all the vectors in g_witness
            // output_wit.vectors[0] now contains decomp(t) || decomp(g)
            output_wit.vectors[0].concat(&mut PolyVec(
                g_witness.into_iter().map(|poly| poly.0).flatten().collect(),
            ));
        }
    }

    let mut hasher = Shake128::default();
    hasher.update(&output_stat.hash);

    if proof.tail {
        // hash inner and garbage
        for byte in inner.iter_bytes().chain(garbage.iter_bytes()) {
            hasher.update(&[byte]);
        }

        output_stat.commitments = Commitments::Tail { inner, garbage };
    } else {
        // hash u1
        let u1_bytes: Vec<u8> = u1.iter_bytes().collect();
        hasher.update(&u1_bytes);

        output_stat.commitments = Commitments::NoTail { inner, u1, u2 };
    }

    // copy commitments from output_stat to proof
    proof.commitments = output_stat.commitments.clone();

    let mut reader = hasher.finalize_xof();

    // store hash in output_stat
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
        inner_commit_tail(&mut inner, &witness, &matrix_a);

        for (chunk, res) in inner.0.chunks_exact(com_rank_1).zip(expected_result.iter()) {
            assert_eq!(chunk, res.0)
        }
    }
}

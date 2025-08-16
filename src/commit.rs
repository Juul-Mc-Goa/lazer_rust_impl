use crate::{
    Seed,
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

impl std::fmt::Debug for CommitKeyData {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use CommitKeyData::*;
        match self {
            Tail {
                matrix_a,
                matrices_b,
                matrices_c,
            } => {
                write!(f, "data: tail\n")?;
                write!(
                    f,
                    "  matrix_a: matrix of size ({}, {})\n",
                    matrix_a.0.len(),
                    matrix_a.0[0].len()
                )?;
                write!(
                    f,
                    "  matrices_b: {} x {} matrices\n",
                    matrices_b.len(),
                    matrices_b[0].len()
                )?;
                write!(
                    f,
                    "  matrices_c: {} x {} matrices\n",
                    matrices_c.len(),
                    matrices_c[0].len()
                )?;
            }
            NoTail {
                matrix_a,
                matrices_b,
                matrices_c,
                matrices_d,
            } => {
                write!(f, "data: no tail\n")?;
                write!(
                    f,
                    "  matrix_a:   size ({}, {})\n",
                    matrix_a.0.len(),
                    matrix_a.0[0].len()
                )?;
                write!(
                    f,
                    "  matrices_b: {} x {} matrices\n",
                    matrices_b.len(),
                    matrices_b[0].len()
                )?;
                write!(
                    f,
                    "  matrices_c: {} x {} matrices\n",
                    matrices_c.len(),
                    matrices_c[0].len()
                )?;
                write!(
                    f,
                    "  matrices_d: {} x {} matrices",
                    matrices_d.len(),
                    matrices_d[0].len()
                )?;
            }
        }

        Ok(())
    }
}

#[allow(dead_code)]
pub struct CommitKey {
    /// The matrices used for commiting.
    pub data: CommitKeyData,
    /// Seed used to generate `self.data`.
    pub seed: <ChaCha8Rng as SeedableRng>::Seed,
}

impl std::fmt::Debug for CommitKey {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "  seed: {:?}\n", self.seed)?;
        write!(f, "  {:?}", self.data)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl CommitKey {
    pub fn new(tail: bool, r: usize, dim: usize, com_params: &CommitParams, seed: Seed) -> Self {
        // generate data
        let mut rng = ChaCha8Rng::from_seed(seed);

        let matrix_a: PolyMatrix = PolyMatrix::random(&mut rng, com_params.commit_rank_1, dim);
        let matrices_b: Vec<Vec<PolyMatrix>> = (0..com_params.uniform_length)
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
            .map(|_k| {
                (0..r)
                    .map(|i| PolyMatrix::random(&mut rng, com_params.commit_rank_2, i + 1))
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
                .map(|_k| {
                    (0..r)
                        .map(|i| PolyMatrix::random(&mut rng, com_params.commit_rank_2, i + 1))
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

    pub fn inner(&self) -> &PolyVec {
        use Commitments::*;
        match self {
            Tail { inner, garbage: _ } => inner,
            NoTail {
                inner,
                u1: _,
                u2: _,
            } => inner,
        }
    }

    pub fn garbage(&self) -> Option<&PolyVec> {
        use Commitments::*;
        match self {
            Tail { inner: _, garbage } => Some(garbage),
            NoTail { .. } => None,
        }
    }

    pub fn outer(&self) -> Option<(&PolyVec, &PolyVec)> {
        use Commitments::*;
        match self {
            Tail { .. } => None,
            NoTail { inner: _, u1, u2 } => Some((u1, u2)),
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
    let last_vec = outer_wit.last_mut().unwrap();

    for i in 0..input_wit.len() {
        let mut chunk = matrix_a.apply(&input_wit[i]);

        // decompose inner_commit
        for (k, inner_decomp) in chunk.decomp(unif_base, unif_len).iter_mut().enumerate() {
            // apply the matrices in matrices_b
            matrices_b[k][i].add_apply(outer, inner_decomp);
            // extend last_vec
            last_vec.concat(inner_decomp);
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
    output_stat: &mut Statement,
    output_wit: &mut Witness,
    proof: &mut Proof,
    input_wit: &Witness,
) {
    let r = input_wit.r;
    let com_params = &output_stat.commit_params;
    let commit_key = &output_stat.commit_key;

    // resized witness
    let mut full_witness = vec![PolyVec(Vec::with_capacity(output_stat.dim)); output_stat.r];

    let mut vector_idx = 0; // in 0..r

    output_stat.commitments = Commitments::new(proof.tail);
    let mut inner = PolyVec::new();
    let mut garbage = PolyVec::new();
    let mut u1 = PolyVec::zero(com_params.commit_rank_2);
    let u2 = PolyVec::zero(com_params.commit_rank_2);

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
            full_witness.resize(
                full_witness.len() + proof.chunks[i] - 1,
                PolyVec::zero(output_stat.dim),
            );

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
    use crate::random_context;

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

    #[test]
    fn commit_tail_dimensions() {
        let r: usize = 5;
        let dim: usize = 5;
        let tail = true;
        let (wit, mut proof, stat) = random_context(r, dim, tail);

        let mut output_stat = Statement::new(&proof, &stat.hash);
        let mut output_wit = Witness::new(&output_stat);

        // commit
        commit(&mut output_stat, &mut output_wit, &mut proof, &wit);

        let com_params = proof.commit_params;
        let Commitments::Tail { inner, garbage } = proof.commitments else {
            panic!("proof.commitments is NoTail");
        };

        // inner commitment is T_1 || ... || T_r: each T_i is
        // t_i (size = com_param_1) || many zeros (size = (z_len - 1) * com_param_1)
        assert_eq!(
            inner.0.len(),
            com_params.z_length * stat.r * com_params.commit_rank_1
        );

        // garbage is r(r+1)/2 coefficients
        let r = output_stat.r;
        assert_eq!(garbage.0.len(), r * (r + 1) / 2);
    }

    #[test]
    fn commit_no_tail_dimensions() {
        let r: usize = 5;
        let dim: usize = 5;
        let tail = false;
        let (wit, mut proof, stat) = random_context(r, dim, tail);

        let mut output_stat = Statement::new(&proof, &stat.hash);
        let mut output_wit = Witness::new(&output_stat);

        // commit
        commit(&mut output_stat, &mut output_wit, &mut proof, &wit);

        let com_params = proof.commit_params;
        let Commitments::NoTail { inner, u1, u2: _ } = proof.commitments else {
            panic!("proof.commitments is NoTail");
        };

        // inner commitment is T_1 || ... || T_r: each T_i is
        // t_i (size = com_rank_1) || many zeros (size = (z_len - 1) * com_rank_1)
        assert_eq!(
            inner.0.len(),
            com_params.z_length * stat.r * com_params.commit_rank_1
        );

        assert_eq!(u1.0.len(), com_params.commit_rank_2);
    }
}

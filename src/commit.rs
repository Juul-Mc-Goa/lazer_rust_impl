use crate::{
    Seed,
    linear_algebra::{PolyMatrix, PolyVec},
    proof::Proof,
    ring::PolyRingElem,
    statement::Statement,
    utils::{add_apply_matrices_b, add_apply_matrices_garbage},
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
#[derive(Clone)]
pub enum CommitKeyData {
    /// The matrices `A, B, C` used to compute
    /// `u1 = B * decomp(A * z) + C * decomp(quad_garbage)`.
    Tail {
        /// matrix of size `commit_rank_1 * dim`
        matrix_a: PolyMatrix,
        /// `z_length * r` matrices of size `commit_rank_2 * commit_rank_1`
        matrices_b: Vec<Vec<PolyMatrix>>,
        /// `quad_length * r` matrices, where `matrix_c[k][i]` has size `commit_rank_2 * (r - i)`.
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

impl CommitKeyData {
    pub fn matrix_a(&self) -> &PolyMatrix {
        use CommitKeyData::*;
        match self {
            Tail { matrix_a, .. } => matrix_a,
            NoTail { matrix_a, .. } => matrix_a,
        }
    }

    pub fn matrices_b(&self) -> &Vec<Vec<PolyMatrix>> {
        use CommitKeyData::*;
        match self {
            Tail {
                matrix_a: _,
                matrices_b,
                ..
            } => matrices_b,
            NoTail {
                matrix_a: _,
                matrices_b,
                ..
            } => matrices_b,
        }
    }

    pub fn matrices_c(&self) -> &Vec<Vec<PolyMatrix>> {
        use CommitKeyData::*;
        match self {
            Tail {
                matrix_a: _,
                matrices_b: _,
                matrices_c,
                ..
            } => matrices_c,
            NoTail {
                matrix_a: _,
                matrices_b: _,
                matrices_c,
                ..
            } => matrices_c,
        }
    }

    pub fn matrices_d(&self) -> Option<&Vec<Vec<PolyMatrix>>> {
        use CommitKeyData::*;
        match self {
            Tail { .. } => None,
            NoTail {
                matrix_a: _,
                matrices_b: _,
                matrices_c: _,
                matrices_d,
            } => Some(matrices_d),
        }
    }
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
#[derive(Clone)]
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
        garbage: PolyVec, // size: 2r
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

    pub fn garbage_mut(&mut self) -> Option<&mut PolyVec> {
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

    pub fn outer_mut(&mut self) -> Option<(&mut PolyVec, &mut PolyVec)> {
        use Commitments::*;
        match self {
            Tail { .. } => None,
            NoTail { inner: _, u1, u2 } => Some((u1, u2)),
        }
    }

    pub fn string_hash(&self) -> String {
        use Commitments::*;
        match self {
            Tail { inner, garbage } => {
                PolyVec::join(&[inner.clone(), garbage.clone()]).string_hash()
            }
            NoTail { inner, u1, u2 } => {
                PolyVec::join(&[inner.clone(), u1.clone(), u2.clone()]).string_hash()
            }
        }
    }
}

/// Commit:
/// - modify `output_stat.commitments`
///   - inner commitment is all `t_i = As_i`
///   - if `proof.tail` is true: `garbage` is all `< s_i, s_j >`
///   - if `proof.tail` is false: `u1 = Bt + Cg`
/// - modify `output_wit`
///   - resize `input_wit.s_i` and append to `output_wit`
///   - push a last element: `t || g || h`
///
/// Return re-packaged witness.
#[allow(dead_code)]
pub fn commit(
    output_stat: &mut Statement,
    output_wit: &mut Witness,
    proof: &mut Proof,
    input_wit: &Witness,
) -> Witness {
    let r = input_wit.r;
    let com_params = &output_stat.commit_params;
    let commit_key = &output_stat.commit_key;

    // resized witness
    let packed_witness = proof.pack_witness(&input_wit);

    output_stat.commitments = Commitments::new(proof.tail);
    let mut inner = PolyVec::new();
    let mut garbage = PolyVec::new();
    let mut u1 = PolyVec::zero(com_params.commit_rank_2);
    let u2 = PolyVec::zero(com_params.commit_rank_2);

    let matrix_a = commit_key.data.matrix_a();
    let matrices_b = commit_key.data.matrices_b();
    let matrices_c = commit_key.data.matrices_c();

    // last vector of output_wit: t || g || h
    let mut tgh = PolyVec::new();

    // step 1: handle inner commitment (t)
    for s_i in &packed_witness.vectors {
        // apply the matrix A
        let mut t_i = matrix_a.apply(s_i);
        inner.concat(&mut t_i);
    }

    if !proof.tail {
        let unif_base = com_params.uniform_base;
        let unif_len = com_params.uniform_length;

        // decompose inner
        let mut t: PolyVec = PolyVec::join(&inner.decomp(unif_base, unif_len));

        // update u1
        add_apply_matrices_b(&mut u1.0, matrices_b, &t.0, r, com_params);
        // update: tgh = t
        tgh.concat(&mut t);
    }

    // step 2: handle garbage (g)
    if com_params.quadratic_length != 0 {
        // compute quadratic garbage < s_i, s_j >
        let mut big_g: Vec<Vec<PolyRingElem>> = Vec::new();
        let s = &packed_witness.vectors;
        for i in 0..output_stat.r {
            big_g.push(vec![PolyRingElem::zero(); i + 1]);
            for j in 0..=i {
                big_g[i][j] += s[i].scalar_prod(&s[j]);
            }
        }

        if proof.tail {
            // append the quadratic garbage to commitments.garbage
            for mut garbage_vec in big_g.drain(..) {
                garbage.0.append(&mut garbage_vec);
            }
        } else {
            // - compute g
            // - apply matrices_c, update u1
            // - append g to tgh
            let quad_base = com_params.quadratic_base;
            let quad_len = com_params.quadratic_length;

            // compute g (= quad_garbage)
            let mut big_g_concat = PolyVec::new();
            big_g
                .into_iter()
                .for_each(|g_i| big_g_concat.concat(&mut PolyVec(g_i)));

            let mut g: PolyVec = PolyVec::new();
            big_g_concat
                .decomp(quad_base, quad_len)
                .iter_mut()
                .for_each(|g_k| g.concat(g_k));

            // update u1
            add_apply_matrices_garbage(&mut u1.0, matrices_c, &g.0, r, quad_len);
            // tgh <- tgh || g
            tgh.concat(&mut g);
        }
    }

    output_wit.vectors.push(tgh);

    let mut hasher = Shake128::default();
    hasher.update(&output_stat.hash);

    if proof.tail {
        // hash inner and garbage
        inner
            .iter_bytes()
            .chain(garbage.iter_bytes())
            .for_each(|byte| {
                hasher.update(&[byte]);
            });

        output_stat.commitments = Commitments::Tail { inner, garbage };
    } else {
        // hash u1
        let u1_bytes: Vec<u8> = u1.iter_bytes().collect();
        hasher.update(&u1_bytes);

        output_stat.commitments = Commitments::NoTail { inner, u1, u2 };
    }

    // copy commitments from output_stat to proof
    proof.commitments = output_stat.commitments.clone();

    // store hash in output_stat
    let mut reader = hasher.finalize_xof();
    reader.read(&mut output_stat.hash);

    return packed_witness;
}

#[cfg(test)]
mod tests {
    use crate::{generate_context, random_seed};

    use super::*;

    #[test]
    fn commit_tail_dimensions() {
        let r: usize = 5;
        let dim: usize = 5;
        let tail = true;

        let seed = random_seed();
        let (wit, mut proof, stat) = generate_context(r, dim, tail, seed);

        let mut output_stat = Statement::new(&proof, &stat.hash, None);
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
        let seed = random_seed();
        let (wit, mut proof, stat) = generate_context(r, dim, tail, seed);

        let mut output_stat = Statement::new(&proof, &stat.hash, None);
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

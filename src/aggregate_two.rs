// TODO: aggregate all zero constraints to a single one

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::{
    commit::{CommitKeyData, CommitParams, Commitments},
    linear_algebra::{PolyMatrix, PolyVec, SparsePolyMatrix},
    proof::Proof,
    ring::{BaseRingElem, PolyRingElem},
    statement::Statement,
};

/// A structure used to a recursed witness `decomp(z) || decomp(t) || decomp(g)
/// || decomp(h)` or (dually) a linear constraint on such a witness.
struct RecursedVector {
    /// Decomposition of `z`: `z_len` PolyVecs of size `dim`.
    z_part: Vec<PolyVec>,
    /// Decomposition of each `t_i`: ` unif_len * r` PolyVecs of size `com_rank_1`.
    t_part: Vec<Vec<PolyVec>>,
    /// Decomposition of each `g_ij` (with `i <= j`): `quad_len` PolyVecs of size `r(r+1)/2`.
    g_part: Vec<PolyVec>,
    /// Decomposition of each `h_ij` (with `i <= j`): `unif_len` PolyVecs of size `r(r+1)/2`.
    h_part: Vec<PolyVec>,
}

#[allow(dead_code)]
impl RecursedVector {
    /// Initialize fields with `PolyVec::zero()`.
    fn new(input_stat: &Statement) -> Self {
        let com_params = &input_stat.commit_params;
        let z_len = input_stat.commit_params.z_length;
        let unif_len = input_stat.commit_params.uniform_length;
        let quad_len = input_stat.commit_params.quadratic_length;
        let dim = input_stat.dim;
        let r = input_stat.r;
        let quad_number = r * (r + 1) / 2;

        let z = vec![PolyVec::zero(dim); z_len];
        let t = vec![vec![PolyVec::zero(com_params.commit_rank_1); r]; unif_len];
        let g = vec![PolyVec::zero(quad_number); quad_len];
        let h = vec![PolyVec::zero(quad_number); unif_len];

        Self {
            z_part: z,
            t_part: t,
            g_part: g,
            h_part: h,
        }
    }

    /// Generate a linear constraint on the split witnesses.
    pub fn to_lin_constraint(self, proof: &Proof) -> PolyVec {
        let split_dim = proof.split_dim;
        let mut result: Vec<PolyVec> = Vec::new();

        // split each z_k into vectors of dimension `split_dim`
        for z_k in self.z_part {
            result.append(&mut z_k.into_chunks(split_dim));
        }

        // t, g, h only appear in linear constraints: we concat them into one
        // vector v
        let mut v: PolyVec = PolyVec::new();
        for t_k in self.t_part {
            v.concat(&mut PolyVec::join(&t_k));
        }
        v.concat(&mut PolyVec::join(&self.g_part));
        v.concat(&mut PolyVec::join(&self.h_part));

        // split v into vectors of dimension `split_dim`
        result.append(&mut v.into_chunks(split_dim));

        PolyVec::join(&result)
    }

    /// Build linear part of the constraint:
    /// `u_1 = sum(ik, B_ik t_ik) + sum(ijk, C_ijk g_ijk)`.
    ///
    /// The constant part (`< c_1, u1 >`) has to be built elsewhere.
    fn add_u1_constraint(
        &mut self,
        c_1: &PolyVec,
        matrices_b: &[Vec<PolyMatrix>],
        matrices_c: &[Vec<PolyMatrix>],
        com_params: &CommitParams,
        input_stat: &Statement,
    ) {
        // update self.t
        for k in 0..com_params.uniform_length {
            for i in 0..input_stat.r {
                self.t_part[k][i] = matrices_b[k][i].apply_transpose(&c_1);
            }
        }

        // update self.g
        for k in 0..com_params.quadratic_length {
            for i in 0..input_stat.r {
                for (j, coord) in matrices_c[k][i].apply_transpose(&c_1).0.iter().enumerate() {
                    let quad_i = i * (i + 1) / 2;
                    self.g_part[k].0[quad_i + j] += coord;
                }
            }
        }
    }

    /// Build linear part of the constraint: `u_2 = sum(ijk, D_ijk h_ijk)`.
    ///
    /// The constant part (`< c_2, u2 >`) has to be built elsewhere.
    fn add_u2_constraint(
        &mut self,
        c_2: &PolyVec,
        matrices_d: &[Vec<PolyMatrix>],
        com_params: &CommitParams,
        input_stat: &Statement,
    ) {
        for k in 0..com_params.uniform_length {
            for i in 0..input_stat.r {
                for (j, poly) in matrices_d[k][i].apply_transpose(&c_2).0.iter().enumerate() {
                    let quad_i: usize = i * (i + 1) / 2;
                    self.h_part[k].0[quad_i + j] += poly;
                }
            }
        }
    }

    /// Build constraint: `Az = sum(i, c_i t_i)`.
    fn add_inner_constraint(
        &mut self,
        c_z: &PolyVec,
        matrix_a: &PolyMatrix,
        com_params: &CommitParams,
        input_stat: &Statement,
        challenges: &[PolyRingElem],
    ) {
        // matrix_part = -transpose(A) * c_z
        let mut matrix_part = matrix_a.apply_transpose(c_z);
        matrix_part.neg();

        let z_base_mod_p: BaseRingElem = (1 << com_params.z_base as u64).into();

        // update self.z
        for k in 0..com_params.z_length {
            self.z_part[k] += &matrix_part;
            matrix_part *= &z_base_mod_p;
        }

        let unif_base_mod_p: BaseRingElem = (1 << com_params.uniform_base as u64).into();

        // update self.t
        for i in 0..input_stat.r {
            let mut t_i = c_z * &challenges[i];

            for k in 0..com_params.uniform_length {
                self.t_part[k][i] += &t_i;
                t_i *= &unif_base_mod_p;
            }
        }
    }

    /// Build linear part of the constraint:
    /// `< z, z > = sum(ijk, ((2 ^ quad_base) ^ k) c_i c_j g_ijk)`.
    ///
    /// The quadratic part (`< z, z, >`) has to be built elsewhere.
    fn add_g_constraint(
        &mut self,
        c_g: &PolyRingElem,
        com_params: &CommitParams,
        input_stat: &Statement,
        challenges: &[PolyRingElem],
    ) {
        let mut k_part = -c_g.clone();
        let quad_base_mod_p: BaseRingElem = (1 << com_params.quadratic_base as u64).into();

        for k in 0..com_params.quadratic_length {
            for i in 0..input_stat.r {
                let i_part = &k_part * &challenges[i];
                for j in 0..=i {
                    let quad_i: usize = i * (i + 1) / 2;
                    self.g_part[k].0[quad_i + j] += &i_part * &challenges[j];
                }
            }

            k_part = k_part * quad_base_mod_p;
        }
    }

    /// Build constraint:
    /// `< sum(i, c_i b_i), z > = sum(ij, c_i c_j h_ij)`.
    fn add_h_constraint(
        &mut self,
        linear_part: &PolyVec,
        com_params: &CommitParams,
        input_stat: &Statement,
        challenges: &[PolyRingElem],
    ) {
        // step 1: < -sum(i, c_i b_i), z >
        // update self.z
        let z_base_mod_p: BaseRingElem = (1 << com_params.z_base as u64).into();
        let linear_part = linear_part.clone().into_chunks(input_stat.dim);
        let mut lin_aggregate = PolyVec::zero(input_stat.dim);
        linear_part
            .iter()
            .zip(challenges.iter())
            .for_each(|(b_i, c_i)| lin_aggregate.add_mul_assign(c_i, b_i));
        lin_aggregate.neg();

        for k in 0..com_params.z_length {
            self.z_part[k] += &lin_aggregate;
            lin_aggregate *= &z_base_mod_p;
        }

        // step 2: sum(ij, c_i c_j h_ij)
        // update self.h
        let unif_base_mod_p: BaseRingElem = (1 << com_params.uniform_base as u64).into();
        let mut k_part = PolyRingElem::one();

        (0..com_params.uniform_length).for_each(|k| {
            (0..input_stat.r).for_each(|i| {
                let quad_i: usize = i * (i + 1) / 2;
                let i_part: PolyRingElem = &k_part * &challenges[i];

                (0..=i).for_each(|j| {
                    self.h_part[k].0[quad_i + j] += &i_part * &challenges[j];
                });
            });

            k_part *= unif_base_mod_p;
        });
    }

    /// Build linear part of the constraint:
    /// `sum(ijk, ((2 ^ quad_base) ^ k) a_ij g_ijk) +
    /// sum(ik, ((2 ^ unif_base) ^ k) h_iik) +
    /// constant = 0`.
    ///
    /// The `constant` part has to be built elsewhere.
    fn add_agg_constraint(
        &mut self,
        c_agg: &PolyRingElem,
        quadratic_part: &SparsePolyMatrix,
        com_params: &CommitParams,
        input_stat: &Statement,
    ) {
        let quad_base_mod_p: BaseRingElem = (1 << com_params.quadratic_base as u64).into();
        let unif_base_mod_p: BaseRingElem = (1 << com_params.uniform_base as u64).into();

        // update self.g
        for (i, j, coef) in quadratic_part.0.iter() {
            let quad_i = i * (i + 1) / 2;
            let mut scaled_coef = c_agg * coef;

            for k in 0..com_params.quadratic_length {
                self.g_part[k].0[quad_i + j] += &scaled_coef;
                scaled_coef = scaled_coef * quad_base_mod_p;
            }
        }

        // update self.h
        for i in 0..input_stat.r {
            let h_ii_idx = i * (i + 1) / 2 + i;
            let mut scaled_coef = c_agg.clone();
            for k in 0..com_params.uniform_length {
                self.h_part[k].0[h_ii_idx] += &scaled_coef;
                scaled_coef = scaled_coef * unif_base_mod_p;
            }
        }
    }
}

/// Build the full constraint in `output_stat`:
/// - this constraint is expressed on the family of vectors obtained by
///   splitting the vector `decomp(z) || decomp(t) || decomp(g) || decomp(h)`,
/// - the following constraints are aggregated:
///   1. `u_1 = sum(ik, B_ik t_ik) + sum(ijk, C_ijk g_ijk)`
///   2. `u_2 = sum(ijk, D_ijk h_ijk)`
///   3. `Az = sum(i, c_i t_i)`
///   4. `< z, z > = sum(ij, g_ij c_i c_j)`
///   5. `sum(i, c_i < lin_part[i], z >) = sum(ij, h_ij c_i c_j)`
///   6. `sum(ij, a_ij g_ij) + sum(i, h_ii) - b = 0`
#[allow(dead_code)]
pub fn aggregate_input_stat(output_stat: &mut Statement, proof: &Proof, input_stat: &Statement) {
    // update output_stat: set
    // witness <- decomp(z) || decomp(t) || decomp(g) || decomp(h)
    // and modify output_stat.constraint accordingly
    //
    // t = t_1 || ... || t_r
    // g = g_11 || g_21 || g_22 || ... || g_rr
    // h = h_11 || h_21 || h_22 || ... || h_rr
    //
    // decompose in a small basis:
    // t_i  = sum(k, (z_base ^ k) t_ik)
    // g_ij = sum(k, (quad_base ^ k) g_ijk)
    // h_ij = sum(k, (unif_base ^ k) h_ijk)
    //
    // 2 * com_rank_2 constraints: one for each coordinate of u_1 || u_2.
    // constraints:
    //   1. u_1 = sum(ik, B_ik t_ik) + sum(ijk, C_ijk g_ijk)
    //   2. u_2 = sum(ijk, D_ijk h_ijk)
    //
    // com_rank_1 constraints: one for each coordinate of inner_commit.
    //   Az = sum(i, c_i t_i)
    //
    // 3 more constraints:
    //   1. < z, z > = sum(ij, g_ij c_i c_j)
    //   2. sum(i, c_i < lin_part[i], z >) = sum(ij, h_ij c_i c_j)
    //   3. sum(ij, a_ij g_ij) + sum(i, h_ii) + c = 0
    //
    // we will generate 2 * com_rank_2 + com_rank_1 + 2 challenges
    //   1. c_1 of rank com_rank_2 (for u_1)
    //   2. c_2 of rank com_rank_2 (for u_2)
    //   3. c_z of rank com_rank_1 (for z)
    //   4. c_g of rank 1   (for extra constraint n°1)
    //   5. c_agg of rank 1 (for extra constraint n°3)
    //
    // extra constraint n°2 has trivial challenge equal to 1

    let com_params = input_stat.commit_params;
    let com_rank1 = com_params.commit_rank_1;
    let com_rank2 = com_params.commit_rank_2;
    let commit_key = &output_stat.commit_key;

    let mut hashbuf = [0_u8; 32];
    hashbuf[..16].copy_from_slice(&output_stat.hash);
    let mut rng = ChaCha8Rng::from_seed(hashbuf);

    let c_1: PolyVec = PolyVec::challenge(com_rank2, &mut rng);
    let c_2: PolyVec = PolyVec::challenge(com_rank2, &mut rng);
    let c_z: PolyVec = PolyVec::challenge(com_rank1, &mut rng);
    let c_g: PolyRingElem = PolyRingElem::challenge(&mut rng);
    let c_agg: PolyRingElem = PolyRingElem::challenge(&mut rng);

    let CommitKeyData::NoTail {
        matrix_a,
        matrices_b,
        matrices_c,
        matrices_d,
    } = &commit_key.data
    else {
        panic!("aggregate 2: commitment key is CommitKeyData::Tail");
    };

    let mut recursed_vector = RecursedVector::new(input_stat);

    let Commitments::NoTail { inner: _, u1, u2 } = &input_stat.commitments else {
        panic!("aggregate 2: commitments in input statement is Commitments::Tail");
    };

    // handle u1:
    recursed_vector.add_u1_constraint(&c_1, matrices_b, matrices_c, &com_params, input_stat);
    output_stat.constraint.constant = -c_1.scalar_prod(u1);

    // handle u2:
    recursed_vector.add_u2_constraint(&c_2, matrices_d, &com_params, input_stat);
    output_stat.constraint.constant -= c_2.scalar_prod(u2);

    // handle z:
    recursed_vector.add_inner_constraint(
        &c_z,
        matrix_a,
        &com_params,
        input_stat,
        &input_stat.challenges,
    );

    // handle h:
    recursed_vector.add_h_constraint(
        &input_stat.constraint.linear_part,
        &com_params,
        input_stat,
        &input_stat.challenges,
    );

    // handle aggregated relation:
    recursed_vector.add_agg_constraint(
        &c_agg,
        &output_stat.constraint.quadratic_part,
        &com_params,
        input_stat,
    );
    output_stat.constraint.constant += &c_agg * &input_stat.constraint.constant;

    // if the quadratic part in `input_stat` is non-empty:
    if !input_stat.constraint.quadratic_part.0.is_empty() {
        // handle g:
        recursed_vector.add_g_constraint(&c_g, &com_params, input_stat, &input_stat.challenges);

        // handle quadratic part:
        //   - z = sum(
        //       k in 0..f,
        //       ((2 ^ z_base) ^ k) z_k
        //     )
        //   - < z, z > = sum(
        //       k in 0..f,
        //       ((2 ^ z_base) ^ 2k) < z_k, z_k >
        //     ) + 2 * sum(
        //       l < k in 0..f,
        //       ((2 ^ z_base) ^ (k + l)) < z_k, z_l >
        //     )
        //   - each z_k is a concatenation of witnesses:
        //     z_k = s_i || ... || s_(i + chunks - 1)

        let chunks = proof.chunks[0];
        let z_base = input_stat.commit_params.z_base;
        let z_length = input_stat.commit_params.z_length;

        output_stat.constraint.quadratic_part.0 =
            Vec::with_capacity(chunks * z_length * (z_length + 1) / 2);

        for k in 0..z_length {
            for l in 0..=k {
                let log_scale_factor = z_base * (k + l);
                let scale_factor: BaseRingElem = (1 << log_scale_factor).into();

                // compute `-< z, z >`: coef is either -1 or -2
                let mut coef: PolyRingElem = if l == k {
                    -PolyRingElem::one()
                } else {
                    let two: BaseRingElem = 2.into();
                    (-two).into()
                };

                coef = &scale_factor * coef;

                for chunk_idx in 0..chunks {
                    output_stat.constraint.quadratic_part.0.push((
                        chunks * k + chunk_idx,
                        chunks * l + chunk_idx,
                        coef.clone(),
                    ));
                }
            }
        }
    }

    // copy recursed vector to linear part of the output constraint
    output_stat.constraint.linear_part = recursed_vector.to_lin_constraint(proof);
}

// TODO: aggregate all zero constraints to a single one

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::{
    commit::{CommitKey, CommitKeyData, CommitParams, Commitments},
    linear_algebra::{PolyMatrix, PolyVec, SparsePolyMatrix},
    ring::{BaseRingElem, PolyRingElem},
    statement::Statement,
};

/// A structure used to a recursed witness `decomp(z) || decomp(t) || decomp(g)
/// || decomp(h)` or (dually) a linear constraint on such a witness.
struct RecursedVector {
    /// Decomposition of `z`: `z_len` PolyVecs of size `dim`.
    z_part: Vec<PolyVec>,
    /// Decomposition of each `t_i`: `unif_len` PolyVecs of size `com_rank_1`.
    t_part: Vec<Vec<PolyVec>>,
    /// Decomposition of each `g_ij`: `quad_len` PolyVecs of size `r(r+1)/2`
    g_part: Vec<PolyVec>,
    /// Decomposition of each `h_ij`: `unif_len` PolyVecs of size `r(r+1)/2`
    h_part: Vec<PolyVec>,
}

#[allow(dead_code)]
impl RecursedVector {
    /// Initialize fields with `PolyVec::zero()`.
    fn new(input_stat: &Statement) -> Self {
        let com_params = &input_stat.commit_params;
        let dim = input_stat.dim;
        let r = input_stat.r;
        let quad_number = r * (r + 1) / 2;

        let z_part = vec![PolyVec::zero(dim); com_params.z_length];
        let t_part =
            vec![vec![PolyVec::zero(com_params.commit_rank_1); dim]; com_params.uniform_length];
        let g_part = vec![PolyVec::zero(quad_number); com_params.quadratic_length];
        let h_part = vec![PolyVec::zero(quad_number); com_params.uniform_length];

        Self {
            z_part,
            t_part,
            g_part,
            h_part,
        }
    }

    /// Build linear part of the constraint:
    /// `u_1 = sum(ik, B_ik t_ik) + sum(ijk, C_ijk g_ijk)`.
    ///
    /// The constant part (`u1`) has to be built elsewhere.
    fn add_u1_constraint(
        &mut self,
        c_1: &PolyVec,
        matrices_b: &[Vec<PolyMatrix>],
        matrices_c: &[Vec<PolyMatrix>],
        com_params: &CommitParams,
        input_stat: &Statement,
    ) {
        // update self.t_part
        for k in 0..com_params.uniform_length {
            for i in 0..input_stat.r {
                self.t_part[k][i] = matrices_b[i][k].apply_transpose(&c_1);
            }
        }

        // update self.g_part
        for k in 0..com_params.quadratic_length {
            for i in 0..input_stat.r {
                for (j, poly) in matrices_c[i][k].apply_transpose(&c_1).0.iter().enumerate() {
                    self.g_part[k].0[i * (i + 1) / 2 + j] += poly;
                }
            }
        }
    }

    /// Build linear part of the constraint: `u_2 = sum(ijk, D_ijk h_ijk)`.
    ///
    /// The constant part (`u2`) has to be built elsewhere.
    fn add_u2_constraint(
        &mut self,
        c_2: &PolyVec,
        matrices_d: &[Vec<PolyMatrix>],
        com_params: &CommitParams,
        input_stat: &Statement,
    ) {
        for k in 0..com_params.uniform_length {
            for i in 0..input_stat.r {
                for (j, poly) in matrices_d[i][k].apply_transpose(&c_2).0.iter().enumerate() {
                    self.h_part[k].0[i * (i + 1) / 2 + j] += poly;
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
    ) {
        // matrix_part = - transpose(A) * c_z
        let mut matrix_part = matrix_a.apply_transpose(c_z);
        matrix_part.0.iter_mut().for_each(|poly| poly.neg());

        let z_base_mod_p: BaseRingElem = (com_params.z_base as u64).into();

        // update self.z_part
        for k in 0..com_params.z_length {
            self.z_part[k].add_assign(&matrix_part);
            matrix_part.mul_assign(z_base_mod_p);
        }

        let unif_base_mod_p: BaseRingElem = (com_params.uniform_base as u64).into();

        // update self.t_part
        for i in 0..input_stat.r {
            let mut t_i_part = c_z.clone();
            t_i_part.mul_assign(input_stat.challenges[i].clone());

            for k in 0..com_params.uniform_length {
                self.t_part[k][i].add_assign(&t_i_part);
                t_i_part.mul_assign(unif_base_mod_p);
            }
        }
    }

    /// Build linear part of the constraint:
    /// `< z, z > = sum(ijk, (quad_base ^ k) c_i c_j g_ijk)`.
    ///
    /// The quadratic part has to be built elsewhere.
    fn add_g_constraint(
        &mut self,
        c_g: &PolyRingElem,
        com_params: &CommitParams,
        input_stat: &Statement,
    ) {
        let mut k_part = -c_g.clone();
        let quad_base_mod_p: BaseRingElem = (com_params.quadratic_base as u64).into();

        for k in 0..com_params.quadratic_length {
            for i in 0..input_stat.r {
                let i_part = &k_part * &input_stat.challenges[i];
                for j in 0..i {
                    self.g_part[k].0[i * (i + 1) / 2 + j] += &i_part * &input_stat.challenges[j];
                }
            }

            k_part = k_part * quad_base_mod_p;
        }
    }

    /// Build constraint:
    /// `sum(i, < c_i linear_part[i], z >) = sum(ij, c_i c_j h_ij)`.
    fn add_h_constraint(
        &mut self,
        linear_part: &[PolyVec],
        com_params: &CommitParams,
        input_stat: &Statement,
    ) {
        let z_base_mod_p: BaseRingElem = (com_params.z_base as u64).into();

        // update self.z_part
        for part in linear_part {
            let mut k_part = part.clone();

            for k in 0..com_params.z_length {
                self.z_part[k].add_assign(&k_part);
                k_part.mul_assign(z_base_mod_p);
            }
        }

        let mut k_part = PolyRingElem::one();
        let unif_base_mod_p: BaseRingElem = (com_params.uniform_base as u64).into();

        // update self.h_part
        for k in 0..com_params.uniform_length {
            for i in 0..input_stat.r {
                let i_part = &k_part * &input_stat.challenges[i];
                for j in 0..i {
                    self.h_part[k].0[i * (i + 1) / 2 + j] += &i_part * &input_stat.challenges[j];
                }
            }

            k_part = k_part * unif_base_mod_p;
        }
    }

    /// Build linear part of the constraint:
    /// `sum(ijk, (quad_base ^ k) a_ij g_ijk) + sum(ik, (unif_base ^ k) h_iik) + constant = 0`.
    ///
    /// The `constant` part has to be built elsewhere.
    fn add_agg_constraint(
        &mut self,
        c_agg: &PolyRingElem,
        quadratic_part: &SparsePolyMatrix,
        com_params: &CommitParams,
        input_stat: &Statement,
    ) {
        let quad_base_mod_p: BaseRingElem = (com_params.quadratic_base as u64).into();
        let unif_base_mod_p: BaseRingElem = (com_params.uniform_base as u64).into();

        // update self.g_part
        for (i, j, coef) in quadratic_part.0.iter() {
            let mut scaled_coef = c_agg * coef;
            for k in 0..com_params.quadratic_length {
                self.g_part[k].0[i * (i + 1) + j] += &scaled_coef;
                scaled_coef = scaled_coef * quad_base_mod_p;
            }
        }

        // update self.h_part
        for i in 0..input_stat.r {
            let mut scaled_coef = c_agg.clone();
            for k in 0..com_params.uniform_length {
                self.h_part[k].0[i * (i + 1) + i] += &scaled_coef;
                scaled_coef = scaled_coef * unif_base_mod_p;
            }
        }
    }
}

pub fn aggregate_input_stat(
    output_stat: &mut Statement,
    input_stat: &Statement,
    commit_key: &CommitKey,
) {
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
    //   3. sum(ij, a_ij g_ij) + sum(i, h_ii) - b = 0
    //
    // we will generate 2 * com_rank_2 + com_rank_1 + 2 challenges
    //   1. c_1 of rank com_rank_2 (for u_1)
    //   2. c_2 of rank com_rank_2 (for u_2)
    //   3. c_z of rank com_rank_1 (for z)
    //   4. c_g of rank 1   (for extra constraint n°1)
    //   5. c_agg of rank 1 (for extra constraint n°3)

    let com_params = input_stat.commit_params;
    let com_rank1 = com_params.commit_rank_1;
    let com_rank2 = com_params.commit_rank_2;

    let mut hashbuf = [0_u8; 32];
    hashbuf[..16].copy_from_slice(&output_stat.hash);
    let mut rng = ChaCha8Rng::from_seed(hashbuf);

    let c_1: PolyVec = PolyVec::random(com_rank2, &mut rng);
    let c_2: PolyVec = PolyVec::random(com_rank2, &mut rng);
    let c_z: PolyVec = PolyVec::random(com_rank1, &mut rng);
    let c_g: PolyRingElem = PolyRingElem::random(&mut rng);
    let c_agg: PolyRingElem = PolyRingElem::random(&mut rng);

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
    output_stat.constraint.constant -= c_1.scalar_prod(u1);

    // handle u2:
    recursed_vector.add_u2_constraint(&c_2, matrices_d, &com_params, input_stat);
    output_stat.constraint.constant -= c_2.scalar_prod(u2);

    // handle z:
    recursed_vector.add_inner_constraint(&c_z, matrix_a, &com_params, input_stat);

    // handle g:
    recursed_vector.add_g_constraint(&c_g, &com_params, input_stat);
    // TODO: build quadratic part

    // handle h:
    recursed_vector.add_h_constraint(&input_stat.constraint.linear_part, &com_params, input_stat);

    // handle aggregated relation:
    recursed_vector.add_agg_constraint(
        &c_agg,
        &output_stat.constraint.quadratic_part,
        &com_params,
        input_stat,
    );
    output_stat.constraint.constant += c_agg * &input_stat.constraint.constant;

    // TODO: handle quadratic part
}

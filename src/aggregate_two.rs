// TODO: aggregate all zero constraints to a single one

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::{
    commit::{CommitKey, CommitKeyData, Commitments},
    linear_algebra::PolyVec,
    proof::Proof,
    ring::PolyRingElem,
    statement::Statement,
};

pub fn aggregate_input_stat(
    output_stat: &mut Statement,
    proof: &Proof,
    input_stat: &Statement,
    commit_key: &CommitKey,
) {
    // merge input_stat with output_stat
    // witness <- witness || decomp(t) || decomp(g) || decomp(h)
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
    //   4. c_g of rank 1 (for extra constraint n°1)
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

    let Commitments::NoTail { inner, u1, u2 } = &input_stat.commitments else {
        panic!("aggregate 2: commitments in input statement is Commitments::Tail");
    };

    // handle u1: sum(ik, B_ik t_ik)
    for i in 0..input_stat.r {
        for k in 0..com_params.z_length {
            output_stat
                .constraint
                .linear_part
                .push(matrices_b[i][k].apply_transpose(&c_1));
        }
    }

    // handle u1: sum(ijk, C_ijk g_ijk)
    let mut quad_garbage: PolyVec = PolyVec::new();
    for i in 0..input_stat.r {
        for k in 0..com_params.quadratic_length {
            quad_garbage.concat(&mut matrices_c[i][k].apply_transpose(&c_1));
        }
    }
    output_stat.constraint.linear_part.push(quad_garbage);

    // handle u1: update constraint.constant
    output_stat.constraint.constant -= c_1.scalar_prod(u1);

    // handle u2: sum(ijk, D_ijk h_ijk)
    let mut lin_garbage: PolyVec = PolyVec::new();
    for i in 0..input_stat.r {
        for k in 0..com_params.uniform_length {
            lin_garbage.concat(&mut matrices_d[i][k].apply_transpose(&c_2));
        }
    }
    output_stat.constraint.linear_part.push(lin_garbage);

    // handle u2: update constraint.constant
    output_stat.constraint.constant -= c_2.scalar_prod(u2);

    // handle z: two parts
    //     < c_3, Az > = < c_3, sum(i, c_i t_i) >
    //     sum(i, c_i < lin_part[i], z >) = sum(ij, c_i c_j h_ij)
    todo!()
    // TODO: handle g
    // TODO: handle h
}

use crate::{commit::CommitParams, proof::Proof, statement::Statement, witness::Witness};

pub fn amortize_tail(
    output_stat: &mut Statement,
    output_wit: &mut Witness,
    proof: &mut Proof,
    tmp_wit: &Witness,
) {
    todo!()
}

/// Transform a witness `(s_1, ..., ,s_r)` into a new witness `(z, t, g, h)`.
/// Here we have:
/// - `z = sum(i, c_i s_i)`
/// - `t` is the decomposition in the base `unif_base` of a vector `t_prime`, where
///   `t_prime` is the concatenation of all `t_i = A * s_i`
/// - `g` is the decomposition in the base `quad_base` of a vector `g_prime`, where
///   `g_prime` has `r(r+1) / 2` coordinates `g_ij = < s_i, s_j >` (where `i <= j`)
/// - `h` is the decomposition in the base `unif_base` of a vector `h_prime`, where
///   `h_prime` has `r(r+1) / 2` coordinates
///   `h_ij = (< b_i, s_j > + < b_j, s_i >) / 2` (where `i <= j`)
pub fn amortize(
    output_stat: &mut Statement,
    output_wit: &mut Witness,
    proof: &mut Proof,
    tmp_wit: &Witness,
) {
    if proof.tail {
        amortize_tail(output_stat, output_wit, proof, tmp_wit);
    }

    let com_params = output_stat.commit_params;
    let Statement {
        r,
        dim,
        dim_inner,
        tail,
        commit_params:
            CommitParams {
                z_base,
                z_length,
                uniform_base: unif_base,
                uniform_length: unif_len,
                quadratic_base: quad_base,
                quadratic_length: quad_len,
                commit_rank_1: com_rank_1,
                commit_rank_2: com_rank_2,
                ..
            },
        commitments,
        challenges,
        constraint,
        squared_norm_bound,
        hash,
    } = output_stat;
    let (r, unif_base, unif_len, quad_base, quad_len, com_rank_1, com_rank_2) = (
        *r,
        *unif_base,
        *unif_len,
        *quad_base,
        *quad_len,
        *com_rank_1,
        *com_rank_2,
    );

    let h_start = r + unif_len * com_rank_1 + quad_len * r * (r + 1) / 2;
    let (_, wit_h) = output_wit.vectors[com_params.z_length]
        .0
        .split_at_mut(h_start);

    todo!()
}

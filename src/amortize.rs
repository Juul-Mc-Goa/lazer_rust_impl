use crate::{
    commit::{CommitKey, CommitKeyData, CommitParams, Commitments},
    linear_algebra::PolyVec,
    proof::Proof,
    ring::PolyRingElem,
    statement::Statement,
    witness::Witness,
};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

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
    commit_key: &CommitKey,
) {
    if proof.tail {
        amortize_tail(output_stat, output_wit, proof, tmp_wit);
    }

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
                quadratic_base: _,
                quadratic_length: _,
                commit_rank_1: _,
                commit_rank_2: com_rank_2,
                ..
            },
        commitments:
            Commitments::NoTail {
                inner: _,
                u1: _,
                u2,
            },
        challenges,
        constraint,
        squared_norm_bound,
        hash,
    } = output_stat
    else {
        panic!("amortize: output statement is `Tail`");
    };
    let (dim, r, z_length, unif_base, unif_len, com_rank_2) =
        (*dim, *r, *z_length, *unif_base, *unif_len, *com_rank_2);

    // compute linear garbage
    let linear_part = &constraint.linear_part;
    let dim_h = unif_len * r * (r + 1) / 2;
    let mut h: PolyVec = PolyVec(Vec::with_capacity(dim_h));

    for k in 0..dim {
        for i in 0..r {
            let coef_ik = &linear_part[i].0[k];
            let wit_ik = &tmp_wit.vectors[i].0[k];
            h.0.push(coef_ik * wit_ik);
            for j in i..r {
                let coef_jk = &linear_part[j].0[k];
                let wit_jk = &tmp_wit.vectors[j].0[k];
                h.0.push(coef_ik * wit_jk + coef_jk * wit_ik);
            }
        }
    }

    // decompose linear garbage
    let mut h = h.decomp(unif_base);

    let CommitKey {
        data:
            CommitKeyData::NoTail {
                matrix_a: _,
                matrices_b: _,
                matrices_c: _,
                matrices_d,
            },
        seed: _,
    } = commit_key
    else {
        panic!()
    };

    // compute u2
    *u2 = PolyVec::zero(com_rank_2);
    for (h_digit, mat_level) in h.iter().zip(matrices_d.iter()) {
        for i in 0..r {
            mat_level[i].add_apply_raw(&mut u2.0, &h_digit.0[i..]);
        }
    }

    // use u2 to update output_stat.hash
    // initialise hasher
    let mut hasher = Shake128::default();
    hasher.update(hash);
    hasher.update(&u2.iter_bytes().collect::<Vec<_>>());
    let mut reader = hasher.finalize_xof();

    // update hash
    let mut hashbuf = [0_u8; 32];
    reader.read(&mut hashbuf);
    hash.copy_from_slice(&hashbuf[..16]);

    // generate challenges
    let mut rng = ChaCha8Rng::from_seed(hashbuf);
    challenges.resize_with(r, || PolyRingElem::challenge(&mut rng));

    // compute z
    let mut z = PolyVec::zero(dim);
    tmp_wit
        .vectors
        .iter()
        .zip(challenges.iter())
        .for_each(|(w, c)| z.add_mul_assign(c, w));

    // decompose z
    let mut z = z.decomp(*z_base);

    // compute output_wit.norm_square[i]
    //         output_stat.norm_bound[i]
    for i in 0..=z_length {
        output_wit.norm_square[i] = output_wit.vectors[i].norm_square();
        *squared_norm_bound += output_wit.norm_square[i];
    }

    proof.norm_square = *squared_norm_bound;
}

use crate::{
    commit::{CommitKey, CommitKeyData, CommitParams, Commitments},
    constants::{DEGREE, PRIME_BYTES_LEN},
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
    let r = output_stat.r;
    let z_base = output_stat.commit_params.z_base;
    let mut h: Vec<PolyRingElem> = Vec::with_capacity(2 * r);
    let mut hashbuf = [0_u8; 16 + 2 * DEGREE as usize * PRIME_BYTES_LEN];

    let phi = &mut output_stat.constraint.linear_part;
    let s = &tmp_wit.vectors;
    let challenges = &mut output_stat.challenges;

    let make_rng_challenge = |buf: &[u8]| -> PolyRingElem {
        let mut seed = [0_u8; 32];
        seed.copy_from_slice(&buf[..32]);
        let mut rng = ChaCha8Rng::from_seed(seed);

        PolyRingElem::challenge(&mut rng)
    };

    // compute < phi_0, s_0 >
    let mut s_acc = s[0].clone();
    let mut phi_acc = phi[0].clone();
    h.push(phi[0].scalar_prod(&s_acc));

    // shake128(output_stat.hash || h[0])
    let mut hasher = Shake128::default();
    hasher.update(&output_stat.hash);
    hasher.update(&h[0].to_le_bytes());
    let mut reader = hasher.finalize_xof();
    reader.read(&mut hashbuf[..32]);

    challenges.push(make_rng_challenge(&hashbuf));

    for i in 1..r {
        // compute h[2i - 1]
        h.push(phi[i].scalar_prod(&s_acc));
        h[2 * i - 1] += phi_acc.scalar_prod(&s[i]);

        // compute h[2i]
        h.push(phi[i].scalar_prod(&s[i]));

        // generate challenges[i]: update hashbuf
        let mut hasher = Shake128::default();
        hasher.update(&hashbuf[0..16]);
        hasher.update(&h[2 * i - 1].to_le_bytes());
        hasher.update(&h[2 * i].to_le_bytes());
        let mut reader = hasher.finalize_xof();
        reader.read(&mut hashbuf);

        // generate challenges[i]
        challenges.push(make_rng_challenge(&hashbuf));

        // update s_acc, phi_acc
        s_acc.add_mul_assign(&challenges[i], &s[i]);
        phi_acc.add_mul_assign(&challenges[i], &phi[i]);
    }

    let Commitments::Tail {
        mut inner,
        mut garbage,
    } = output_stat.commitments
    else {
        panic!("amortize tail: output_stat.commitments is tail");
    };

    garbage.0.append(&mut h);

    // store new hash
    output_stat.hash.copy_from_slice(&hashbuf[..16]);

    // compute z
    //         output_wit.norm_square
    //         output_stat.squared_norm_bound
    let mut new_wit: Vec<PolyVec> = Vec::new();
    for (i, z_small) in s_acc.decomp(z_base).into_iter().enumerate() {
        output_wit.norm_square[i] = z_small.norm_square();
        output_stat.squared_norm_bound += output_wit.norm_square[i];
        new_wit.push(z_small);
    }

    new_wit.append(&mut output_wit.vectors);
    output_wit.vectors = new_wit;

    proof.norm_square = output_stat.squared_norm_bound;
    phi.resize(output_stat.dim, PolyVec::new());
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
        dim_inner: _,
        tail: _,
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
    let h = h.decomp(unif_base);

    // concatenate all PolyVecs in h
    let mut h_concat: PolyVec = PolyVec::new();
    for h_small in h.iter() {
        h_concat.concat(&mut h_small.clone());
    }

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
    let mut z: Vec<PolyVec> = z.decomp(*z_base);

    // build new witness: z part
    let mut new_wit: Vec<PolyVec> = Vec::with_capacity(r);
    new_wit.append(&mut z);

    // build new witness: g || h part
    let last_idx = z_length;
    new_wit.push(output_wit.vectors[0].clone());
    new_wit[last_idx].concat(&mut h_concat);

    // store new witness
    output_wit.vectors = new_wit;

    for i in 0..=z_length {
        output_wit.norm_square[i] = output_wit.vectors[i].norm_square();
        *squared_norm_bound += output_wit.norm_square[i];
    }

    proof.norm_square = *squared_norm_bound;
}

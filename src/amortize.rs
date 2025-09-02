use crate::{
    commit::{CommitKey, CommitKeyData, Commitments},
    constants::{DEGREE, PRIME_BYTES_LEN},
    linear_algebra::PolyVec,
    proof::Proof,
    ring::PolyRingElem,
    statement::Statement,
    utils::add_apply_matrices_garbage,
    witness::Witness,
};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

/// Amortize when `proof.tail` is true.
pub fn amortize_tail(
    output_stat: &mut Statement,
    input_stat: &Statement,
    output_wit: &mut Witness,
    proof: &mut Proof,
    tmp_wit: &Witness,
) {
    let r = output_stat.r;
    let dim = output_stat.dim;
    let z_base = output_stat.commit_params.z_base;
    let z_len = output_stat.commit_params.z_length;
    let mut h: Vec<PolyRingElem> = Vec::with_capacity(2 * r);
    let mut hashbuf = [0_u8; 16 + 2 * DEGREE as usize * PRIME_BYTES_LEN];

    let linear_part = &input_stat.constraint.linear_part;
    let s = &tmp_wit.vectors;
    let challenges = &mut output_stat.challenges;

    // compute h_0 = < phi_0, s_0 >
    let mut s_acc = s[0].clone();
    let mut phi_acc = PolyVec(linear_part.0[..dim].to_vec());
    h.push(phi_acc.scalar_prod(&s_acc));

    // shake128(output_stat.hash || h[0])
    let mut hasher = Shake128::default();
    hasher.update(&output_stat.hash);
    hasher.update(&h[0].to_le_bytes());
    let mut reader = hasher.finalize_xof();
    reader.read(&mut hashbuf[..32]);

    challenges.push(PolyRingElem::challenge_from_seed(&hashbuf[..32]));
    s_acc *= &challenges[0];
    phi_acc *= &challenges[0];

    // tail garbage: h_i for i in 0..2r
    // h_(2i - 1) = sum(j < i, c_j (< phi[i], s_j > + < phi[j], s_i >))
    // h_(2i) = < phi[i], s_i >
    for i in 1..r {
        // compute h[2i - 1]
        let (start, end) = (dim * i, dim * (i + 1));
        let phi_i = PolyVec(linear_part.0[start..end].to_vec());
        h.push(phi_i.scalar_prod(&s_acc));
        h[2 * i - 1] += phi_acc.scalar_prod(&s[i]);

        // compute h[2i]
        h.push(phi_i.scalar_prod(&s[i]));

        // generate challenges[i]: update hashbuf
        let mut hasher = Shake128::default();
        hasher.update(&hashbuf[0..16]);
        hasher.update(&h[2 * i - 1].to_le_bytes());
        hasher.update(&h[2 * i].to_le_bytes());
        let mut reader = hasher.finalize_xof();
        reader.read(&mut hashbuf);

        // generate challenges[i]
        challenges.push(PolyRingElem::challenge_from_seed(&hashbuf[..32]));

        // update s_acc, phi_acc
        s_acc.add_mul_assign(&challenges[i], &s[i]);
        phi_acc.add_mul_assign(&challenges[i], &phi_i);
    }

    let Commitments::Tail {
        inner: _,
        ref mut garbage,
    } = output_stat.commitments
    else {
        panic!("amortize tail: output_stat.commitments is NoTail");
    };

    garbage.0.append(&mut h);

    let Commitments::Tail {
        inner: _,
        garbage: proof_garbage,
    } = &mut proof.commitments
    else {
        panic!("amortize_tail(): proof.commitments should be Tail.");
    };

    *proof_garbage = garbage.clone();

    // store new hash
    output_stat.hash.copy_from_slice(&hashbuf[..16]);

    // compute:
    //   z
    //   output_wit.norm_square
    //   output_stat.squared_norm_bound
    let mut new_wit: Vec<PolyVec> = Vec::new();

    output_wit.norm_square.resize(z_len + 1, 0_u128);

    for (i, z_small) in s_acc.decomp(z_base, z_len).into_iter().enumerate() {
        output_wit.norm_square[i] = z_small.norm_square();
        output_stat.squared_norm_bound += output_wit.norm_square[i];
        new_wit.push(z_small);
    }

    new_wit.append(&mut output_wit.vectors);
    output_wit.vectors = new_wit;
    output_wit.dim = output_wit.vectors.iter().map(|w| w.0.len()).collect();
    proof.norm_square = output_stat.squared_norm_bound;
    output_stat
        .constraint
        .linear_part
        .0
        .resize(dim * r, PolyRingElem::zero());
}

/// Transform a witness `(s_1, ..., ,s_r)` into a new witness `(z, t, g, h)`.
/// Here we have:
/// - `z = sum(i, c_i s_i)`
/// - `t` is the decomposition in the base `unif_base` of a vector `t_prime`, where
///   `t_prime` is the concatenation of all `t_i = A * s_i`
/// - `g` is the decomposition in the base `quad_base` of a vector `g_prime`, where
///   `g_prime` has `r(r+1) / 2` coordinates `g_ij = < s_i, s_j >` (where `j <= i`)
/// - `h` is the decomposition in the base `unif_base` of a vector `h_prime`, where
///   `h_prime` has `r(r+1) / 2` coordinates
///   `h_ij = < b_i, s_j > + < b_j, s_i >` (where `j <= i`)
pub fn amortize(
    output_stat: &mut Statement,
    input_stat: &Statement,
    output_wit: &mut Witness,
    proof: &mut Proof,
    packed_wit: &Witness,
) {
    if proof.tail {
        amortize_tail(output_stat, input_stat, output_wit, proof, packed_wit);
    }

    let (r, dim, squared_norm_bound, hash, commit_key) = (
        output_stat.r,
        output_stat.dim,
        &mut output_stat.squared_norm_bound,
        &mut output_stat.hash,
        &output_stat.commit_key,
    );

    let (z_base, z_len, unif_base, unif_len, com_rank_2) = (
        output_stat.commit_params.z_base,
        output_stat.commit_params.z_length,
        output_stat.commit_params.uniform_base,
        output_stat.commit_params.uniform_length,
        output_stat.commit_params.commit_rank_2,
    );

    let Commitments::NoTail {
        inner: _,
        u1: _,
        u2,
    } = &mut output_stat.commitments
    else {
        panic!("wrong `Tail` variant for output statement's commitments");
    };

    let (challenges, constraint) = (&input_stat.challenges, &input_stat.constraint);

    // compute linear garbage
    let linear_part = &constraint.linear_part;
    let dim_h = unif_len * r * (r + 1) / 2;
    let mut h: PolyVec = PolyVec(Vec::with_capacity(dim_h));

    for i in 0..r {
        let lin_i = linear_part.clone_range(dim * i, dim * (i + 1));
        for j in 0..i {
            let lin_j = linear_part.clone_range(dim * j, dim * (j + 1));
            h.0.push(
                lin_i.scalar_prod(&packed_wit.vectors[j])
                    + lin_j.scalar_prod(&packed_wit.vectors[i]),
            );
        }
        // diagonal coef is special
        h.0.push(lin_i.scalar_prod(&packed_wit.vectors[i]));
    }

    // decompose and concatenate linear garbage
    let mut h = PolyVec::join(&h.decomp(unif_base, unif_len));

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
        panic!("amortize: commit key is `Tail`")
    };

    // compute u2
    *u2 = PolyVec::zero(com_rank_2);
    // sum(j <= i, D_ijk h_ijk)
    add_apply_matrices_garbage(&mut u2.0, matrices_d, &h.0, r, unif_len);

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
    output_stat.challenges = (0..r).map(|_| PolyRingElem::challenge(&mut rng)).collect();

    // compute z
    let mut z = PolyVec::zero(dim);

    challenges
        .iter()
        .zip(packed_wit.vectors.iter())
        .for_each(|(c_i, s_i)| z.add_mul_assign(c_i, s_i));

    // decompose z
    let mut z: Vec<PolyVec> = z.decomp(z_base, z_len);

    // new_witness = (z_0, ... z_{z_len - 1}, t || g || h)
    // build new witness: z part
    let mut new_wit: Vec<PolyVec> = Vec::with_capacity(r);
    new_wit.append(&mut z);

    // build new_witness: t || g || h
    // t || g is stored in output_wit[0] (computed in `commit()`)
    let last_idx = z_len;
    new_wit.push(output_wit.vectors[0].clone());
    new_wit[last_idx].concat(&mut h);

    // store new witness
    output_wit.vectors = new_wit;

    for i in 0..=z_len {
        output_wit
            .norm_square
            .push(output_wit.vectors[i].norm_square());
        *squared_norm_bound += output_wit.norm_square[i];
    }

    proof.norm_square = *squared_norm_bound;
}

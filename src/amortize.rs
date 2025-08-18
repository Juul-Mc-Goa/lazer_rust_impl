use crate::{
    commit::{CommitKey, CommitKeyData, CommitParams, Commitments},
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

pub fn amortize_tail(
    output_stat: &mut Statement,
    output_wit: &mut Witness,
    proof: &mut Proof,
    tmp_wit: &Witness,
) {
    let r = output_stat.r;
    let z_base = output_stat.commit_params.z_base;
    let z_len = output_stat.commit_params.z_length;
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
        inner: _,
        ref mut garbage,
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
    for (i, z_small) in s_acc.decomp(z_base, z_len).into_iter().enumerate() {
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
    packed_wit: &Witness,
) {
    if proof.tail {
        amortize_tail(output_stat, output_wit, proof, packed_wit);
    }

    let Statement {
        r,
        dim,
        dim_inner: _,
        tail: _,
        commit_params:
            CommitParams {
                z_base,
                z_length: z_len,
                uniform_base: unif_base,
                uniform_length: unif_len,
                quadratic_base: _,
                quadratic_length: quad_len,
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
        commit_key,
    } = output_stat
    else {
        panic!("amortize: output statement is `Tail`");
    };
    let (dim, r, z_len, unif_base, unif_len, com_rank_2) =
        (*dim, *r, *z_len, *unif_base, *unif_len, *com_rank_2);

    // compute linear garbage
    let linear_part = &constraint.linear_part;
    let dim_h = unif_len * r * (r + 1) / 2;
    println!("amortize: dim_h = {dim_h}");
    let mut h: PolyVec = PolyVec(Vec::with_capacity(dim_h));

    for i in 0..r {
        for j in 0..i {
            h.0.push(
                linear_part[i].scalar_prod(&packed_wit.vectors[j])
                    + linear_part[j].scalar_prod(&packed_wit.vectors[i]),
            );
        }
        // diagonal coef is special
        h.0.push(linear_part[i].scalar_prod(&packed_wit.vectors[i]));
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

    // println!("amortize: u2 = {u2:?}");

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
    challenges
        .iter()
        .zip(packed_wit.vectors.iter())
        .for_each(|(c_i, s_i)| z.add_mul_assign(c_i, s_i));

    println!("z: {z:?}");

    let CommitKeyData::NoTail { ref matrix_a, .. } = output_stat.commit_key.data else {
        panic!("wtf");
    };
    let mut diff: PolyVec = matrix_a.apply(&z);
    diff.neg(); // diff = -Az
    // diff += sum(i, c_i t_i)
    let com_rank_1 = output_stat.commit_params.commit_rank_1;
    let t = output_wit
        .vectors
        .last()
        .unwrap()
        .clone()
        .split(unif_len * r * com_rank_1)
        .0
        .recompose(unif_base as u64, unif_len);
    challenges
        .iter()
        .zip(t.into_chunks(com_rank_1).iter())
        .for_each(|(c_i, t_i)| diff.add_mul_assign(c_i, t_i));

    println!("amortize: diff = {diff:?}");

    // decompose z
    let mut z: Vec<PolyVec> = z.decomp(*z_base, z_len);

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

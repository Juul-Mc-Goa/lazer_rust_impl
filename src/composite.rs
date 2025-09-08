use aes::cipher::KeyIvInit;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

use crate::{
    aggregate_one::collaps_jl_matrices,
    aggregate_two::RecursedVector,
    commit::{CommitKeyData, Commitments},
    constants::{
        CHALLENGE_NORM, DEGREE, JL_MAX_NORM_SQ, LOG_PRIME, PRIME_BYTES_LEN, SLACK, U128_LEN,
    },
    constraint::Constraint,
    jl_matrix::JLMatrix,
    linear_algebra::PolyVec,
    project::proj_norm_square,
    proof::{Proof, sis_secure},
    prove,
    ring::{BaseRingElem, PolyRingElem},
    statement::Statement,
    utils::Aes128Ctr64LE,
    verify::{VerifyResult, verify},
    witness::Witness,
};

pub struct Composite {
    pub l: usize,
    pub size: f64,
    pub proof: Vec<Proof>,
    pub witness: Witness,
}

pub type TempStatement = [Statement; 2];
pub type TempWitness = [Witness; 2];

fn proof_size(proof: &Proof) -> f64 {
    let com_params = proof.commit_params;

    let proj_norm_sq: u128 = proj_norm_square(&proof.projection);

    let mut result = (proj_norm_sq as f64).sqrt();
    result = (result.log2() - 4.0 + 2.05) * 256.0;
    result += (com_params.u1_len + com_params.u2_len + U128_LEN) as f64
        * DEGREE as f64
        * LOG_PRIME as f64;

    result / 8192.0
}

pub fn witness_size(witness: &Witness) -> f64 {
    let mut result: f64 = 0.0;
    let deg_f: f64 = DEGREE as f64;

    for i in 0..witness.r {
        result += (witness.norm_square[i] as f64 / (deg_f * witness.dim[i] as f64))
            .log2()
            .max(0.0)
            * deg_f
            * witness.dim[i] as f64;
    }

    result / 8192.0
}

pub fn composite_prove(
    comp_data: &mut Composite,
    temp_stat: &mut TempStatement,
    temp_wit: &mut TempWitness,
    temp_wit_size: &mut [f64; 2],
) {
    let mut i: usize = 1;
    let mut new_proof: Proof;

    (temp_stat[1], temp_wit[1], new_proof) = prove(&temp_stat[0], &temp_wit[0], false);
    comp_data.proof.push(new_proof);
    temp_wit_size[1] = witness_size(&temp_wit[1]);
    comp_data.l += 1;

    println!("initial proof done");

    println!("Statement 1:",);
    temp_stat[1].print();

    println!("Witness 1:",);
    temp_wit[1].print();

    // iterate until the new proof is bigger than the old one
    // `NoTail` variant is used.
    while comp_data.l < 16 {
        let l = comp_data.l;
        println!("\ncomposite prove {l}");

        (temp_stat[i ^ 1], temp_wit[i ^ 1], new_proof) = prove(&temp_stat[i], &temp_wit[i], false);
        temp_wit[i ^ 1].post_process();

        comp_data.proof.push(new_proof);

        let proof_size = proof_size(&comp_data.proof[l]);
        temp_wit_size[i ^ 1] = witness_size(&temp_wit[i ^ 1]);

        let new_size = proof_size + temp_wit_size[i ^ 1];
        println!("new size: {new_size}, old size: {}", temp_wit_size[i]);
        if proof_size + temp_wit_size[i ^ 1] >= temp_wit_size[i] {
            let _ = comp_data.proof.pop();
            break;
        }

        comp_data.size += proof_size;
        comp_data.l += 1;
        i ^= 1;
    }

    // last proof: `Tail` variant is used.
    if comp_data.l < 16 {
        println!("\n(last proof) composite prove {}", comp_data.l);
        (temp_stat[i ^ 1], temp_wit[i ^ 1], new_proof) = prove(&temp_stat[i], &temp_wit[i], true);

        let proof_size = proof_size(&new_proof);
        comp_data.proof.push(new_proof);

        temp_wit_size[i ^ 1] = witness_size(&temp_wit[i ^ 1]);

        comp_data.size += proof_size;
        comp_data.l += 1;
        i ^= 1;
        // if proof_size + temp_wit_size[i ^ 1] >= temp_wit_size[i] {
        //     println!("(verify) last proof too big: removed");
        //     let _ = comp_data.proof.pop();
        // } else {
        //     comp_data.size += proof_size;
        //     comp_data.l += 1;
        //     i ^= 1;
        // }
    }

    comp_data.witness = temp_wit[i].clone();
    comp_data.size += temp_wit_size[i];
}

fn reduce_commit_tail(output_stat: &mut Statement, proof: &Proof) {
    let Commitments::Tail { inner, garbage } = &proof.commitments else {
        panic!("reduce_commit: proof.commitments should be Tail")
    };
    output_stat.commitments = Commitments::Tail {
        inner: inner.clone(),
        garbage: PolyVec::new(),
    };

    let mut hasher = Shake128::default();
    hasher.update(&output_stat.hash);

    let r = output_stat.r;
    let quad_garbage = garbage.0.split_at(r * (r + 1) / 2).0;

    // hash inner and quad garbage
    // for byte in inner.iter_bytes().chain(garbage.iter_bytes()) {
    for byte in inner
        .iter_bytes()
        .chain(PolyVec::iter_bytes_raw(quad_garbage))
    {
        hasher.update(&[byte]);
    }

    // store hash in output_stat
    let mut reader = hasher.finalize_xof();
    reader.read(&mut output_stat.hash);
}

fn reduce_commit(output_stat: &mut Statement, proof: &Proof) {
    let (proof_u1, _) = proof
        .commitments
        .outer()
        .expect("reduce_commit: proof should be NoTail.");
    output_stat.commitments = Commitments::NoTail {
        inner: PolyVec::new(),
        u1: proof_u1.clone(),
        u2: PolyVec::new(),
    };

    let mut hasher = Shake128::default();
    hasher.update(&output_stat.hash);
    hasher.update(&proof_u1.iter_bytes().collect::<Vec<_>>());
    let mut reader = hasher.finalize_xof();

    reader.read(&mut output_stat.hash);
}

fn reduce_project(
    output_stat: &mut Statement,
    proof: &Proof,
    r: usize,
    bound_square: u128,
) -> Vec<JLMatrix> {
    let upper_bound = JL_MAX_NORM_SQ.min(bound_square);

    if proj_norm_square(&proof.projection) > 256 * upper_bound {
        panic!("reduce_project: Witness projection too big.")
    }

    let mut hasher = Shake128::default();
    hasher.update(&output_stat.hash);
    let mut reader = hasher.finalize_xof();

    let mut hashbuf = [0_u8; 32];
    reader.read(&mut hashbuf);

    // split the result in two
    let (hashbuf_chunks, []) = hashbuf.as_chunks::<16>() else {
        panic!("project: hashbuf length is not a multiple of 16.");
    };
    let hashbuf_left = hashbuf_chunks[0];
    let hashbuf_right = hashbuf_chunks[1];

    let mut cipher =
        Aes128Ctr64LE::new(&hashbuf_right.into(), &proof.jl_nonce.to_le_bytes().into());

    let mut hasher = Shake128::default();
    hasher.update(&hashbuf_left);
    hasher.update(
        &proof
            .projection
            .iter()
            .map(|e| e.to_le_bytes())
            .flatten()
            .collect::<Vec<_>>(),
    );
    let mut reader = hasher.finalize_xof();

    reader.read(&mut output_stat.hash);

    // JL Matrices
    let mut jl_matrices: Vec<JLMatrix> = Vec::new();
    for i in 0..r {
        jl_matrices.push(JLMatrix::random(proof.dim[i], &mut cipher));
    }

    jl_matrices
}

pub fn reduce_gen_lifting_poly(
    output_stat: &mut Statement,
    proof: &Proof,
    i: usize,
    constraint: &mut Constraint,
) {
    // compute result (over R_p) of linear map
    let mut constant = constraint.constant.clone();
    let _c0 = constant.element[0];
    constant = proof.lifting_poly[i].clone();
    constant.element[0] = 0.into();

    constraint.constant = constant;

    // update hash
    let mut hashbuf = [0_u8; 16 + DEGREE as usize * PRIME_BYTES_LEN];
    hashbuf[0..16].copy_from_slice(&output_stat.hash);
    hashbuf[16..].copy_from_slice(&proof.lifting_poly[i].to_le_bytes());

    let mut hasher = Shake128::default();
    hasher.update(&hashbuf);
    let mut reader = hasher.finalize_xof();
    let mut new_hashbuf = [0_u8; 64];
    reader.read(&mut new_hashbuf);

    output_stat.hash.copy_from_slice(&hashbuf[..16]);

    // generate challenge (alpha)
    let alpha: PolyRingElem = PolyRingElem::challenge_from_seed(&new_hashbuf[32..]);

    if i == 0 {
        output_stat.constraint.linear_part = &constraint.linear_part * &alpha;
        output_stat.constraint.constant = &constraint.constant * &alpha;
    } else {
        output_stat.constraint.linear_part += &constraint.linear_part * &alpha;
        output_stat.constraint.constant += &constraint.constant * &alpha;
    }
}

pub fn reduce_agg_amortize(output_stat: &mut Statement, proof: &Proof, input_stat: &Statement) {
    let mut hashbuf = [0_u8; 32];
    hashbuf[..16].copy_from_slice(&output_stat.hash);
    let mut rng = ChaCha8Rng::from_seed(hashbuf);

    let (com_params, com_rank_1, com_rank_2, z_base, z_len) = (
        &output_stat.commit_params,
        output_stat.commit_params.commit_rank_1,
        output_stat.commit_params.commit_rank_2,
        output_stat.commit_params.z_base,
        output_stat.commit_params.z_length,
    );

    if !sis_secure(
        com_rank_1,
        6.0 * CHALLENGE_NORM as f64
            * *SLACK
            * (1 << (z_len - 1) * z_base) as f64
            * (output_stat.squared_norm_bound as f64).sqrt(),
    ) {
        panic!("reduce_amortize(): Inner commitments are not secure");
    }

    if !proof.tail
        && !sis_secure(
            com_rank_2,
            2.0 * *SLACK * (output_stat.squared_norm_bound as f64).sqrt(),
        )
    {
        panic!("reduce_amortize(): Outer commitments are not secure");
    }

    let (squared_norm_bound, commit_key) =
        (&mut output_stat.squared_norm_bound, &output_stat.commit_key);

    *squared_norm_bound = proof.norm_square;

    let c_1: PolyVec = PolyVec::challenge(com_rank_2, &mut rng);
    let c_2: PolyVec = PolyVec::challenge(com_rank_2, &mut rng);
    let c_z: PolyVec = PolyVec::challenge(com_rank_1, &mut rng);
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

    let mut new_linear_part =
        RecursedVector::new(&output_stat.commit_params, input_stat.r, input_stat.dim);

    let Commitments::NoTail { inner: _, u1, u2 } = &input_stat.commitments else {
        panic!("aggregate 2: commitments in input statement is Commitments::Tail");
    };

    // handle u1:
    new_linear_part.add_u1_constraint(&c_1, matrices_b, matrices_c, &com_params, input_stat);
    output_stat.constraint.constant = -c_1.scalar_prod(u1);

    // handle u2:
    new_linear_part.add_u2_constraint(&c_2, matrices_d, &com_params, input_stat);
    output_stat.constraint.constant -= c_2.scalar_prod(u2);

    // handle z:
    new_linear_part.add_inner_constraint(
        &c_z,
        matrix_a,
        &com_params,
        input_stat,
        &input_stat.challenges,
    );

    // handle h:
    new_linear_part.add_h_constraint(
        &input_stat.constraint.linear_part,
        &com_params,
        input_stat,
        &input_stat.challenges,
    );

    // handle aggregated relation:
    new_linear_part.add_agg_constraint(
        &c_agg,
        &output_stat.constraint.quadratic_part,
        &com_params,
        input_stat,
    );
    output_stat.constraint.constant += &c_agg * &input_stat.constraint.constant;

    // if the quadratic part in `input_stat` is non-empty:
    if !input_stat.constraint.quadratic_part.0.is_empty() {
        // handle g
        new_linear_part.add_g_constraint(&c_g, &com_params, input_stat, &input_stat.challenges);

        // handle quadratic part
        let chunks = proof.chunks[0];
        let z_base = input_stat.commit_params.z_base;
        let z_len_in = input_stat.commit_params.z_length;

        output_stat.constraint.quadratic_part.0 =
            Vec::with_capacity(chunks * z_len_in * (z_len_in + 1) / 2);

        for k in 0..z_len_in {
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

    // copy new linear part to output constraint
    output_stat.constraint.linear_part = new_linear_part.to_lin_constraint(proof);
}
pub fn reduce_amortize_tail(output_stat: &mut Statement, proof: &Proof) {
    let r = output_stat.r;
    let com_params = &output_stat.commit_params;
    let (z_base, z_len) = (com_params.z_base, com_params.z_length);

    output_stat.squared_norm_bound = proof.norm_square;

    if !sis_secure(
        com_params.commit_rank_1,
        6.0 * CHALLENGE_NORM as f64
            * *SLACK
            * (1 << (z_len - 1) * z_base) as f64
            * (output_stat.squared_norm_bound as f64).sqrt(),
    ) {
        panic!("reduce_amortize_tail(): Inner commitments are not secure");
    }

    if !proof.tail
        && !sis_secure(
            com_params.commit_rank_2,
            2.0 * *SLACK * (output_stat.squared_norm_bound as f64).sqrt(),
        )
    {
        panic!("reduce_amortize_tail(): Outer commitments are not secure");
    }

    let (
        Commitments::Tail {
            inner: _,
            garbage: stat_garbage,
        },
        Commitments::Tail {
            inner: _,
            garbage: proof_garbage,
        },
    ) = (&mut output_stat.commitments, &proof.commitments)
    else {
        panic!("reduce_amortize_tail(): incompatible Tail variants (output_stat, proof)");
    };

    *stat_garbage = proof_garbage.clone();

    output_stat.challenges = vec![PolyRingElem::zero(); r];

    // garbage is (g_ij (j <= i in 0..r)) || (h_i (i in 0..2r))
    let h = PolyVec(proof_garbage.0.split_at(r * (r + 1) / 2).1.to_vec());
    let mut hashbuf = [0_u8; 16 + 2 * DEGREE as usize * PRIME_BYTES_LEN];

    // generate challenge[0]
    let mut hasher = Shake128::default();
    hasher.update(&output_stat.hash);
    hasher.update(&h.0[0].to_le_bytes());
    let mut reader = hasher.finalize_xof();
    reader.read(&mut hashbuf[..32]);

    output_stat.challenges[0] = PolyRingElem::challenge_from_seed(&hashbuf[..32]);

    (1..r).for_each(|i| {
        // generate challenges[i]: update hashbuf
        let mut hasher = Shake128::default();
        hasher.update(&hashbuf[0..16]);
        hasher.update(&h.0[2 * i - 1].to_le_bytes());
        hasher.update(&h.0[2 * i].to_le_bytes());
        let mut reader = hasher.finalize_xof();
        reader.read(&mut hashbuf);

        // generate challenges[i]
        output_stat.challenges[i] = PolyRingElem::challenge_from_seed(&hashbuf[..32]);
    });

    output_stat.hash.copy_from_slice(&hashbuf[..16]);
}

#[allow(dead_code)]
pub fn reduce_amortize(output_stat: &mut Statement, proof: &Proof) {
    let r = output_stat.r;
    let com_params = &output_stat.commit_params;
    let (z_base, z_len) = (com_params.z_base, com_params.z_length);

    output_stat.squared_norm_bound = proof.norm_square;

    if !sis_secure(
        com_params.commit_rank_1,
        6.0 * CHALLENGE_NORM as f64
            * *SLACK
            * (1 << (z_len - 1) * z_base) as f64
            * (output_stat.squared_norm_bound as f64).sqrt(),
    ) {
        panic!("reduce_amortize(): Inner commitments are not secure");
    }

    if !proof.tail
        && !sis_secure(
            com_params.commit_rank_2,
            2.0 * *SLACK * (output_stat.squared_norm_bound as f64).sqrt(),
        )
    {
        panic!("reduce_amortize(): Outer commitments are not secure");
    }

    use Commitments::*;
    match (&mut output_stat.commitments, &proof.commitments) {
        (
            Tail {
                inner: _,
                garbage: stat_garbage,
            },
            Tail {
                inner: _,
                garbage: proof_garbage,
            },
        ) => *stat_garbage = proof_garbage.clone(),
        (
            NoTail {
                inner: _,
                u1: _,
                u2: stat_u2,
            },
            NoTail {
                inner: _,
                u1: _,
                u2: proof_u2,
            },
        ) => *stat_u2 = proof_u2.clone(),
        _ => panic!("reduce_amortize(): incompatible Tail variants (output_stat, proof)"),
    }

    if proof.tail {
        output_stat.challenges = vec![PolyRingElem::zero(); r];

        // garbage is h_i (i in 0..2r)
        let h = proof.commitments.garbage().unwrap();

        // generate challenge[0]
        let mut seed = [0_u8; 32];
        let mut hasher = Shake128::default();
        hasher.update(&output_stat.hash);
        hasher.update(&h.0[0].to_le_bytes());
        output_stat.challenges[0] = PolyRingElem::challenge_from_seed(&seed);

        (1..r).for_each(|i| {
            // generate challenge[i]
            PolyRingElem::hash_many(&h.0[(2 * i - 1)..(2 * i + 1)], &mut seed);
            output_stat.challenges[i] = PolyRingElem::challenge_from_seed(&seed);
        });
    } else {
        let (_, u2) = output_stat.commitments.outer().unwrap();

        // initialise hasher
        let mut hasher = Shake128::default();
        hasher.update(&output_stat.hash);
        hasher.update(&u2.iter_bytes().collect::<Vec<_>>());
        let mut reader = hasher.finalize_xof();

        // update hash
        let mut hashbuf = [0_u8; 32];
        reader.read(&mut hashbuf);
        output_stat.hash.copy_from_slice(&hashbuf[..16]);

        // generate challenges
        let mut rng = ChaCha8Rng::from_seed(hashbuf);
        output_stat.challenges = (0..r).map(|_| PolyRingElem::challenge(&mut rng)).collect();
    }
    // TODO: lin_part[0] *= challenges[0]
    // for i in 1..r:
    //   lin_part[0] += lin_part[i] * challenges[i]
}

/// Process one step of reduction for composite proofs. This is part of the verifying process
/// for such proofs. See also [`composite_verify`].
pub fn reduce(proof: &Proof, input_stat: &Statement) -> Statement {
    // copy seed from previous statement
    let seed = input_stat.commit_key.seed.clone();
    let mut output_stat: Statement = Statement::new(proof, &input_stat.hash, Some(seed));

    reduce_commit(&mut output_stat, proof);
    println!("(commit) hash: {:?}", output_stat.hash);
    let jl_matrices = reduce_project(
        &mut output_stat,
        proof,
        proof.r,
        input_stat.squared_norm_bound,
    );
    println!("(project) hash: {:?}", output_stat.hash);

    for i in 0..U128_LEN {
        let mut constraint = collaps_jl_matrices(&mut output_stat, proof, &jl_matrices);
        reduce_gen_lifting_poly(&mut output_stat, proof, i, &mut constraint);
    }
    println!("(agg 1) hash: {:?}", output_stat.hash);

    reduce_agg_amortize(&mut output_stat, proof, input_stat);
    println!("(amortize) hash: {:?}", output_stat.hash);

    output_stat
}

pub fn reduce_tail(proof: &Proof, input_stat: &Statement) -> Statement {
    // copy seed from previous statement
    let seed = input_stat.commit_key.seed.clone();
    let mut output_stat: Statement = Statement::new(proof, &input_stat.hash, Some(seed));

    reduce_commit_tail(&mut output_stat, proof);
    println!("(commit tail) hash: {:?}", output_stat.hash);
    // `project` does not depend on proof.tail
    let jl_matrices = reduce_project(
        &mut output_stat,
        proof,
        proof.r,
        input_stat.squared_norm_bound,
    );
    println!("(project tail) hash: {:?}", output_stat.hash);

    for i in 0..U128_LEN {
        let mut constraint = collaps_jl_matrices(&mut output_stat, proof, &jl_matrices);
        reduce_gen_lifting_poly(&mut output_stat, proof, i, &mut constraint);
    }
    println!("(aggregate tail) hash: {:?}", output_stat.hash);

    reduce_amortize_tail(&mut output_stat, proof);
    println!("(amortize tail) hash: {:?}", output_stat.hash);

    output_stat
}

/// Verify a composite proof.
pub fn composite_verify(comp_data: &Composite, statement: &mut [Statement; 2]) -> VerifyResult {
    let mut i: usize = 0;
    let l = comp_data.l;

    // the first l-1 reductions are NoTail
    for j in 0..(l - 1) {
        statement[i ^ 1] = reduce(&comp_data.proof[j], &statement[i]);
        i ^= 1;
    }

    // last reduce, Tail variant
    statement[i ^ 1] = reduce_tail(&comp_data.proof[l - 1], &statement[i]);

    verify(&statement[i ^ 1], &statement[i], &comp_data.witness)
}

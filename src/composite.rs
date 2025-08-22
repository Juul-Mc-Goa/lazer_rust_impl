use aes::cipher::KeyIvInit;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

use crate::{
    aggregate_one::collaps_jl_matrices,
    aggregate_two::aggregate_input_stat,
    commit::Commitments,
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
    l: usize,
    size: f64,
    proof: Vec<Proof>,
    witness: Witness,
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

fn witness_size(witness: &Witness) -> f64 {
    let mut result: f64 = 0.0;
    let deg_f: f64 = DEGREE as f64;

    for i in 0..witness.r {
        result += (witness.norm_square[i] as f64 / (deg_f * witness.dim[i] as f64)).log2()
            * deg_f
            * witness.dim[i] as f64;
    }

    result / 8192.0
}

pub fn composite_prove(
    mut comp_data: Composite,
    mut temp_stat: TempStatement,
    mut temp_wit: TempWitness,
    mut temp_wit_size: [f64; 2],
) {
    let mut i: usize = 0;
    let mut new_proof: Proof;

    // iterate until the new proof is bigger than the old one
    // `NoTail` variant is used.
    while comp_data.l < 16 {
        let l = comp_data.l;

        (temp_stat[i ^ 1], temp_wit[i ^ 1], new_proof) = prove(&temp_stat[i], &temp_wit[i], false);
        comp_data.proof.push(new_proof);

        let proof_size = proof_size(&comp_data.proof[l]);
        temp_wit_size[i ^ 1] = witness_size(&temp_wit[i ^ 1]);

        if proof_size + temp_wit_size[i ^ 1] >= temp_wit_size[i] {
            let _ = comp_data.proof.pop();
            // TODO: remove temp_wit[i^1], temp_stat[i^1] ?
            break;
        }

        comp_data.size += proof_size;
        comp_data.l += 1;
        i ^= 1;
    }

    // last proof: `Tail` variant is used.
    if comp_data.l < 16 {
        (temp_stat[i ^ 1], temp_wit[i ^ 1], new_proof) = prove(&temp_stat[i], &temp_wit[i], true);
        let proof_size = proof_size(&new_proof);
        temp_wit_size[i ^ 1] = witness_size(&temp_wit[i ^ 1]);

        if proof_size + temp_wit_size[i ^ 1] >= temp_wit_size[i] {
            let _ = comp_data.proof.pop();
        } else {
            comp_data.size += proof_size;
            comp_data.l += 1;
            i ^= 1;
        }
    }

    comp_data.witness = temp_wit[i].clone();
    comp_data.size += temp_wit_size[i];
}

fn reduce_commit(output_stat: &mut Statement, proof: &Proof) {
    let (proof_u1, _) = proof
        .commitments
        .outer()
        .expect("reduce commit: proof should be NoTail.");
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
            .map(|e| e.element.to_le_bytes())
            .collect::<Vec<_>>()
            .as_flattened(),
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

pub fn reduce_lift_agg(
    output_stat: &mut Statement,
    proof: &Proof,
    i: usize,
    constraint: &mut Constraint,
) {
    let c0: BaseRingElem = constraint.constant.element[0];
    let mut b: PolyRingElem = proof.lifting_poly[i].clone();
    b.element[0] = c0;
    constraint.constant = b;

    // update hash
    let mut hasher = Shake128::default();

    hasher.update(&output_stat.hash);
    hasher.update(&proof.lifting_poly[i].to_le_bytes());

    let mut reader = hasher.finalize_xof();
    let mut hashbuf = [0_u8; 32];
    reader.read(&mut hashbuf);

    output_stat.hash.copy_from_slice(&hashbuf[..16]);
    let alpha = PolyRingElem::challenge_from_seed(&hashbuf);

    if i == 0 {
        output_stat.constraint.constant = &constraint.constant * &alpha;
        output_stat.constraint.linear_part = &constraint.linear_part * &alpha;
        output_stat.constraint.quadratic_part = &constraint.quadratic_part * &alpha;
    } else {
        output_stat.constraint.constant += &constraint.constant * &alpha;
        output_stat.constraint.linear_part += &constraint.linear_part * &alpha;
        output_stat.constraint.quadratic_part.add_mul_assign(&alpha);
    }
}

pub fn reduce_gen_lifting_poly(
    output_stat: &mut Statement,
    proof: &Proof,
    i: usize,
    constraint: &mut Constraint,
) {
    // compute result (over R_p) of linear map
    let mut constant = constraint.constant.clone();
    let c0 = constant.element[0];
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

pub fn reduce_amortize(output_stat: &mut Statement, proof: &Proof) {
    let (r, dim) = (output_stat.r, output_stat.dim);
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
            com_params.commit_rank_1,
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
        let garbage = proof.commitments.garbage().unwrap();

        // generate challenge[0]
        let mut seed = [0_u8; 32];
        garbage.0[0].hash(&mut seed);
        output_stat.challenges[0] = PolyRingElem::challenge_from_seed(&seed);

        (1..r).for_each(|i| {
            // generate challenge[i]
            PolyRingElem::hash_many(&garbage.0[(2 * i - 1)..(2 * i + 1)], &mut seed);
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
    let jl_matrices = reduce_project(
        &mut output_stat,
        proof,
        proof.r,
        input_stat.squared_norm_bound,
    );

    for i in 0..U128_LEN {
        let mut constraint = collaps_jl_matrices(&mut output_stat, proof, &jl_matrices);
        reduce_gen_lifting_poly(&mut output_stat, proof, i, &mut constraint);
    }

    aggregate_input_stat(&mut output_stat, proof, input_stat);
    reduce_amortize(&mut output_stat, proof);

    output_stat
}

/// Verify a composite proof.
pub fn composite_verify(comp_data: &Composite, statement: &mut [Statement; 2]) -> VerifyResult {
    let mut i: usize = 0;

    for j in 1..comp_data.l {
        statement[i ^ 1] = reduce(&comp_data.proof[j], &statement[i]);
        i ^= 1;
    }

    verify(&statement[i], &statement[i ^ 1], &comp_data.witness)
}

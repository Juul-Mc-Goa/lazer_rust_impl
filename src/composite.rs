use aes::cipher::KeyIvInit;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

use crate::{
    aggregate_one::collaps_jl_matrices,
    aggregate_two::aggregate_two,
    constants::{
        CHALLENGE_NORM, DEGREE, JL_MAX_NORM_SQ, LOG_PRIME, PRIME_BYTES_LEN, SLACK, U128_LEN,
    },
    constraint::Constraint,
    jl_matrix::JLMatrix,
    linear_algebra::PolyVec,
    project::proj_norm_square,
    proof::{Proof, sis_secure},
    prove,
    ring::PolyRingElem,
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
    let mut i: usize = 0;
    let mut new_proof: Proof;

    println!("\n------------------------------------------------------------",);
    println!("Witness 0:");
    temp_wit[0].print();

    // iterate until the new proof is bigger than the old one
    // `NoTail` variant is used.
    while comp_data.l < 16 {
        let l = comp_data.l;
        println!("\n------------------------------------------------------------",);
        println!("composite prove {l}");

        (temp_stat[i ^ 1], temp_wit[i ^ 1], new_proof) = prove(&temp_stat[i], &temp_wit[i], false);

        temp_wit[i ^ 1].post_process();
        println!("Witness {}:", i ^ 1);
        temp_wit[i ^ 1].print();
        println!("\nProof {}", i ^ 1);
        new_proof.print();
        println!("\nStatement {}", i ^ 1);
        temp_stat[i ^ 1].print();

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
        println!("\n------------------------------------------------------------",);
        println!("(last proof) composite prove {}", comp_data.l);
        (temp_stat[i ^ 1], temp_wit[i ^ 1], new_proof) = prove(&temp_stat[i], &temp_wit[i], true);

        println!("Witness {}:", i ^ 1);
        temp_wit[i ^ 1].print();
        println!("\nProof {}", i ^ 1);
        new_proof.print();
        println!("\nStatement {}", i ^ 1);
        temp_stat[i ^ 1].print();

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

fn reduce_commit(output_stat: &mut Statement, proof: &Proof) {
    // copy commitments from proof
    output_stat.commitments = proof.commitments.clone();

    // update hash
    let mut hasher = Shake128::default();
    hasher.update(&output_stat.hash);

    if proof.tail {
        // tail: hash inner and garbage
        let inner = proof.commitments.inner();
        let garbage = proof.commitments.garbage().unwrap();

        inner
            .iter_bytes()
            .chain(garbage.iter_bytes())
            .for_each(|byte| {
                hasher.update(&[byte]);
            });
    } else {
        // no tail: hash u1
        let (u1, _) = proof.commitments.outer().unwrap();
        let u1_bytes: Vec<u8> = u1.iter_bytes().collect();
        hasher.update(&u1_bytes);
    }

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
    let c0 = constraint.constant.element[0];
    let mut constant = proof.lifting_poly[i].clone();
    constant.element[0] = c0;
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

    output_stat.hash.copy_from_slice(&new_hashbuf[..16]);

    // generate challenge (alpha)
    let alpha: PolyRingElem = PolyRingElem::challenge_from_seed(&new_hashbuf[32..]);

    let out_constraint = &mut output_stat.constraint;
    if i == 0 {
        out_constraint.linear_part = &constraint.linear_part * &alpha;
        out_constraint.constant = &constraint.constant * &alpha;
        out_constraint.quadratic_part = &out_constraint.quadratic_part * &alpha;
    } else {
        out_constraint.linear_part += &constraint.linear_part * &alpha;
        out_constraint.constant += &constraint.constant * &alpha;
        out_constraint.quadratic_part =
            &out_constraint.quadratic_part * &(PolyRingElem::one() + &alpha);
    }
}

pub fn reduce_amortize(output_stat: &mut Statement, proof: &Proof) {
    let r = output_stat.r;
    let dim = output_stat.dim;
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
        println!("commit rank 1: {}", com_params.commit_rank_1);
        println!("z_len: {z_len}");
        println!("z_base: {z_base}",);
        println!(
            "log2(proof norm): {}",
            (proof.norm_square as f64).sqrt().log2()
        );
        println!(
            "log2(output_stat norm): {}",
            (output_stat.squared_norm_bound as f64).sqrt().log2()
        );
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

    use crate::Commitments::*;
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

    let mut hasher = Shake128::default();
    hasher.update(&output_stat.hash);

    let mut hashbuf = [0_u8; 16 + 2 * DEGREE as usize * PRIME_BYTES_LEN];

    if proof.tail {
        output_stat.challenges = vec![PolyRingElem::zero(); r];

        // garbage is h_i (i in 0..2r)
        let h = proof.commitments.garbage().unwrap();

        // generate challenge[0]
        hasher.update(&h.0[0].to_le_bytes());
        let mut reader = hasher.finalize_xof();
        reader.read(&mut hashbuf[..32]);

        output_stat.challenges[0] = PolyRingElem::challenge_from_seed(&hashbuf[..32]);

        (1..r).for_each(|i| {
            // update hashbuf
            let mut hasher = Shake128::default();
            hasher.update(&hashbuf[0..16]);
            hasher.update(&h.0[2 * i - 1].to_le_bytes());
            hasher.update(&h.0[2 * i].to_le_bytes());
            let mut reader = hasher.finalize_xof();
            reader.read(&mut hashbuf);

            // generate challenges[i]
            output_stat.challenges[i] = PolyRingElem::challenge_from_seed(&hashbuf[..32]);
        });
    } else {
        let (_, u2) = proof.commitments.outer().unwrap();

        // feed hasher
        hasher.update(&u2.iter_bytes().collect::<Vec<_>>());
        let mut reader = hasher.finalize_xof();

        // update hash
        reader.read(&mut hashbuf[..32]);
        output_stat.hash.copy_from_slice(&hashbuf[..16]);

        // generate challenges
        let mut seed = [0_u8; 32];
        seed.copy_from_slice(&hashbuf[..32]);
        let mut rng = ChaCha8Rng::from_seed(seed);
        output_stat.challenges = (0..r).map(|_| PolyRingElem::challenge(&mut rng)).collect();
    }

    output_stat.hash.copy_from_slice(&hashbuf[..16]);

    // compute new linear part
    let mut phi = PolyVec::zero(dim);

    for i in 0..r {
        let (start, end) = (dim * i, dim * (i + 1));
        phi.add_mul_assign(
            &output_stat.challenges[i],
            &PolyVec(output_stat.constraint.linear_part.0[start..end].to_vec()),
        );
    }

    output_stat.constraint.linear_part = phi;
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

    aggregate_two(&mut output_stat, proof, input_stat);

    reduce_amortize(&mut output_stat, proof);
    println!("(amortize) hash: {:?}", output_stat.hash);

    output_stat
}

/// Verify a composite proof.
pub fn composite_verify(comp_data: &Composite, statement: &mut [Statement; 2]) -> VerifyResult {
    let mut i: usize = 0;
    let l = comp_data.l;

    // the first l-1 reductions are NoTail
    for j in 0..l {
        println!("\n------------------------------------------------------------",);
        println!("composite verify {j}");
        println!("Proof {j}");
        comp_data.proof[j].print();
        println!("\nStatement {i}",);
        statement[i].print();

        statement[i ^ 1] = reduce(&comp_data.proof[j], &statement[i]);
        i ^= 1;
    }

    println!("\n------------------------------------------------------------",);
    println!("verify last...",);
    println!("Witness:",);
    comp_data.witness.print();
    println!("\nStatement {i}",);
    statement[i].print();
    verify(&statement[i], &comp_data.witness)
}

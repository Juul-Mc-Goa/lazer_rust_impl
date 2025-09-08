use std::time::Instant;

use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use crate::{
    aggregate_one::aggregate_constant_coeff,
    aggregate_two::amortize_aggregate,
    commit::{CommitKey, Commitments, commit},
    composite::{Composite, composite_prove, composite_verify, witness_size},
    constraint::Constraint,
    linear_algebra::{PolyVec, SparsePolyMatrix},
    project::project,
    proof::Proof,
    ring::PolyRingElem,
    statement::Statement,
    witness::Witness,
};

#[macro_use]
extern crate lazy_static;

mod constants;
mod utils;

mod constraint;
mod jl_matrix;
mod linear_algebra;
mod proof;
mod ring;
mod statement;
mod witness;

mod aggregate_one;
mod aggregate_two;
mod amortize;
mod commit;
mod dachshund;
mod project;
mod verify;

mod composite;

type Seed = <ChaCha8Rng as SeedableRng>::Seed;

/// Generate seed with `std::rand`.
#[allow(dead_code)]
fn random_seed() -> Seed {
    let mut seed: <ChaCha8Rng as SeedableRng>::Seed = Default::default();
    rand::rng().fill(&mut seed);

    seed
}

/// Generate random context using `seed`.
fn generate_context(r: usize, dim: usize, tail: bool, seed: Seed) -> (Witness, Proof, Statement) {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // let mut rng = ChaCha8Rng::from_os_rng();
    let mut rng = ChaCha8Rng::from_seed(seed.clone());

    let mut wit_vectors: Vec<PolyVec> = Vec::new();
    let mut norm_square: Vec<u128> = Vec::new();
    let mut wit_dim: Vec<usize> = Vec::new();
    let mut squared_norm_bound: u128 = 0;

    for _ in 0..r {
        // let new_s = PolyVec::random(dim, &mut rng);
        let new_s = PolyVec(
            (0..dim)
                .map(|_| PolyRingElem::challenge(&mut rng))
                .collect(),
        );

        wit_dim.push(dim);

        let current_norm_square = new_s.norm_square();
        squared_norm_bound += current_norm_square;
        norm_square.push(current_norm_square);
        wit_vectors.push(new_s);
    }

    let witness = Witness {
        r,
        dim: wit_dim,
        norm_square,
        vectors: wit_vectors,
    };

    let proof = Proof::new(witness.clone(), 1, tail);

    let mut constant = PolyRingElem::zero();

    // build random quadratic part
    let mut quadratic_part = SparsePolyMatrix::new();
    let non_zero_len = 3;

    let mut left = (0..r).collect::<Vec<_>>();
    left.shuffle(&mut rng);
    left.resize(non_zero_len, 0);

    let mut right = (0..r).collect::<Vec<_>>();
    right.shuffle(&mut rng);
    right.resize(non_zero_len, 0);

    for (l, r) in left.into_iter().zip(right.into_iter()) {
        let coef = PolyRingElem::random(&mut rng);
        constant += &coef * witness.vectors[l].scalar_prod(&witness.vectors[r]);
        let i = l.max(r);
        let j = l.min(r);
        quadratic_part.0.push((i, j, coef));
    }

    // build random linear part
    //       random challenges
    let mut linear_part: PolyVec = PolyVec::new();
    let mut challenges: Vec<PolyRingElem> = Vec::new();
    for i in 0..r {
        let mut lin_part_i = PolyVec::random(dim, &mut rng);
        constant += lin_part_i.scalar_prod(&witness.vectors[i]);
        linear_part.concat(&mut lin_part_i);

        challenges.push(PolyRingElem::challenge(&mut rng));
    }

    constant = -constant;
    let commit_params = proof.commit_params.clone();
    let commit_key = CommitKey::new(tail, r, dim, &commit_params, seed);
    let dim_inner = r * commit_params.uniform_length * commit_params.commit_rank_1;

    let constraint = Constraint {
        degree: 1,
        quadratic_part,
        linear_part,
        constant,
    };
    (
        witness,
        proof,
        Statement {
            r,
            dim,
            dim_inner,
            tail,
            commit_params,
            commitments: Commitments::new(tail),
            challenges,
            constraint,
            squared_norm_bound,
            hash: [0; 16],
            commit_key,
        },
    )
}

pub fn prove(statement: &Statement, witness: &Witness, tail: bool) -> (Statement, Witness, Proof) {
    let mut proof = Proof::new(witness.clone(), 1, tail);
    let mut output_stat = Statement::new(
        &proof,
        &statement.hash,
        Some(statement.commit_key.seed.clone()),
    );
    let mut output_wit = Witness::new(&output_stat);

    let now = Instant::now();
    commit(&mut output_stat, &mut output_wit, &mut proof, &witness);
    println!(
        "{:<11}: {:.6} sec, {}",
        "(commit)",
        now.elapsed().as_secs_f32(),
        output_stat.format_hash()
    );

    let packed_wit = proof.pack_witness(&witness);

    let now = Instant::now();

    let jl_matrices = project(&mut output_stat, &mut proof, &witness);

    println!(
        "{:<11}: {:.6} sec, {}",
        "(project)",
        now.elapsed().as_secs_f32(),
        output_stat.format_hash()
    );

    let now = Instant::now();

    aggregate_constant_coeff(&mut output_stat, &mut proof, &witness, &jl_matrices);
    println!(
        "{:<11}: {:.6} sec, {}",
        "(aggregate)",
        now.elapsed().as_secs_f32(),
        output_stat.format_hash()
    );

    let now = Instant::now();
    amortize_aggregate(
        &mut output_stat,
        &mut output_wit,
        &mut proof,
        &packed_wit,
        &statement,
    );
    println!(
        "{:<11}: {:.6} sec, {}",
        "(amortize)",
        now.elapsed().as_secs_f32(),
        output_stat.format_hash()
    );

    (output_stat, output_wit, proof)
}

fn main() {
    {
        let tail = false;
        let (wit, _proof, stat) = generate_context(30, 50, tail, random_seed());
        println!("Generated random context");

        let mut comp_data = Composite {
            l: 0,
            size: 0.0,
            proof: Vec::new(),
            witness: wit.clone(),
        };
        let mut temp_stat = [stat.clone(), stat.clone()];
        let mut temp_wit_size = [witness_size(&wit), 0_f64];
        let mut temp_wit = [wit.clone(), wit];

        composite_prove(
            &mut comp_data,
            &mut temp_stat,
            &mut temp_wit,
            &mut temp_wit_size,
        );

        println!("\nComposite Verify: ");
        let mut temp_stat = [stat.clone(), stat];
        let result = composite_verify(&comp_data, &mut temp_stat);
        println!("\nResult: {result:?}");
    }

    // {
    //     use crate::verify::verify;

    //     let tail = false;
    //     let dim = 128;
    //     let r = 32;
    //     let (wit, _proof, stat) = generate_context(r, dim, tail, random_seed());

    //     println!("Generated random context");

    //     let (output_stat, output_wit, _proof) = prove(&stat, &wit, tail);

    //     print!("\nVerify: ");
    //     let result = verify(&output_stat, &stat, &output_wit);
    //     println!("{result:?}");
    // }
}

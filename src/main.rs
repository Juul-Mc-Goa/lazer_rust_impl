use rand::{Rng, SeedableRng, seq::IteratorRandom};
use rand_chacha::ChaCha8Rng;

use crate::{
    aggregate_one::aggregate_constant_coeff,
    aggregate_two::aggregate_input_stat,
    amortize::amortize,
    commit::{CommitKey, Commitments, commit},
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
mod recursive_prover;
mod verify;

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
    let left = (0..r).choose_multiple(&mut rng, 10);
    let right = (0..r).choose_multiple(&mut rng, 10);

    for (l, r) in left.into_iter().zip(right.into_iter()) {
        let coef = PolyRingElem::random(&mut rng);
        constant += &coef * witness.vectors[l].scalar_prod(&witness.vectors[r]);
        quadratic_part.0.push((l, r, coef));
    }

    // build random linear part
    //       random challenges
    let mut linear_part: Vec<PolyVec> = Vec::new();
    let mut challenges: Vec<PolyRingElem> = Vec::new();
    for i in 0..r {
        let v = PolyVec::random(dim, &mut rng);
        constant += v.scalar_prod(&witness.vectors[i]);
        linear_part.push(v);

        challenges.push(PolyRingElem::challenge(&mut rng));
    }

    constant = -constant;
    let commit_params = proof.commit_params.clone();
    let commit_key = CommitKey::new(tail, r, dim, &commit_params, seed);
    let dim_inner = r * commit_params.uniform_length * commit_params.commit_rank_1;

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
            constraint: Constraint {
                degree: 1,
                quadratic_part: quadratic_part,
                linear_part: linear_part,
                constant: constant,
            },
            squared_norm_bound,
            hash: [0; 16],
            commit_key,
        },
    )
}

fn main() {
    let (wit, mut proof, stat) = generate_context(10, 30, false, Default::default());

    println!("Generated random context");
    println!(
        "  predicted witness norm: {} (ie around 2^{})",
        proof.norm_square,
        proof.norm_square.ilog2()
    );
    println!("  split dimension: {}", proof.split_dim);

    let mut output_stat = Statement::new(&proof, &stat.hash, None);
    let mut output_wit = Witness::new(&output_stat);

    // commit
    commit(&mut output_stat, &mut output_wit, &mut proof, &wit);
    println!("Committed");
    let packed_wit = proof.pack_witness(&wit);
    println!("new witness:");
    packed_wit.print();

    let jl_matrices = project(&mut output_stat, &mut proof, &wit);
    println!("Projected");

    let dim = output_stat.dim;
    aggregate_constant_coeff(&mut output_stat, &mut proof, &wit, dim, &jl_matrices);
    aggregate_input_stat(&mut output_stat, &proof, &stat);
    println!("Aggregated");

    amortize(&mut output_stat, &mut output_wit, &mut proof, &packed_wit);

    println!("\nOutput statement:");
    output_stat.print();
}

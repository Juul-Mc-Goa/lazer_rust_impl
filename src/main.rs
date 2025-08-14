use rand::seq::IteratorRandom;

use crate::{
    commit::{Commitments, commit},
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

fn random_context(r: usize, dim: usize, tail: bool) -> (Witness, Proof, Statement) {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::from_os_rng();

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
    let mut linear_part: Vec<PolyVec> = Vec::new();
    for i in 0..r {
        let v = PolyVec::random(dim, &mut rng);
        constant += v.scalar_prod(&witness.vectors[i]);
        linear_part.push(v);
    }

    constant = -constant;
    let commit_params = proof.commit_params.clone();
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
            challenges: Vec::new(),
            constraint: Constraint {
                degree: 1,
                quadratic_part: quadratic_part,
                linear_part: linear_part,
                constant: constant,
            },
            squared_norm_bound,
            hash: [0; 16],
        },
    )
}

fn main() {
    let (wit, mut proof, stat) = random_context(10, 30, false);

    println!("generated random witness, proof, statement");
    println!("Predicted witness norm: {}", proof.norm_square);

    let (mut output_stat, commit_key) = Statement::new(&proof, &stat.hash);
    let mut output_wit = Witness::new(&output_stat);

    // commit
    commit(
        &commit_key,
        &mut output_stat,
        &mut output_wit,
        &mut proof,
        &wit,
    );

    println!("committed");

    project(&mut output_stat, &mut proof, wit);
}

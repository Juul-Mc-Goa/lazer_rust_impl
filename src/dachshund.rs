use std::time::Instant;

use crate::{
    Witness,
    aggregate_one::collaps_jl_matrices,
    commit::{CommitKey, CommitParams},
    constants::{DEGREE, DEGREE_SIZE, LOG_PRIME, OFFSET, SLACK, U128_LEN},
    linear_algebra::{PolyVec, SparsePolyMatrix},
    project::project,
    proof::Proof,
    ring::{BaseRingElem, PolyRingElem},
    statement::Statement,
};

pub struct SparseConstraint {
    def: usize,
    matrix: SparsePolyMatrix,
    non_zero_len: usize,
    index: Vec<usize>,
    offset: Vec<usize>,
    len: Vec<usize>,
    mult: Vec<usize>,
    linear_part: PolyVec,
    constant: PolyRingElem,
}

pub struct SimpleCommitment {
    pub r: usize,
    pub dim: Vec<usize>,
    pub unif_len: usize,
    pub unif_base: usize,
    pub com_rank_1: usize,
    pub com_rank_2: usize,
    pub u: PolyVec,
    pub alpha: PolyVec,
}

impl SimpleCommitment {
    pub fn new(com_params: &CommitParams, witness: &Witness) -> Self {
        Self {
            r: witness.r,
            dim: witness.dim.clone(),
            unif_len: com_params.uniform_length,
            unif_base: com_params.uniform_base,
            com_rank_1: com_params.commit_rank_1,
            com_rank_2: com_params.commit_rank_2,
            u: PolyVec(Vec::with_capacity(com_params.commit_rank_2)),
            alpha: PolyVec(Vec::with_capacity(witness.r)),
        }
    }
}

pub struct SimpleStatement {
    /// Number of vectors in witness.
    pub r: usize,
    /// Lengths of each vector (`dim` length is `r`).
    pub dim: Vec<usize>,
    /// Number of sparse linear constraints (length is `r`).
    pub constraints_len: usize,
    /// Array of sparse constraints.
    pub constraints: Vec<SparseConstraint>,
    /// Squared norm bounds (length is `r`).
    pub squared_norm_bound: Vec<u128>,
    /// Hash.
    pub hash: [u8; 16],
    /// ???
    pub u0: PolyRingElem,
    /// ???
    pub alpha: PolyRingElem,
    /// commitment matrices
    pub commit_key: CommitKey,
}

impl SimpleStatement {
    pub fn new(
        r: usize,
        dim: Vec<usize>,
        squared_norm_bound: Vec<u128>,
        constraints_len: usize,
        seed: &[u8; 32],
        com_params: &CommitParams,
    ) -> Self {
        let max_dim: usize = dim.iter().fold(0, |prev, d| prev.max(*d));

        Self {
            r,
            dim,
            constraints_len,
            constraints: Vec::new(),
            squared_norm_bound,
            hash: [0_u8; 16],
            u0: PolyRingElem::zero(),
            alpha: PolyRingElem::zero(),
            commit_key: CommitKey::new(false, r, max_dim, com_params, seed.clone()),
        }
    }
}

/// Decompose `a` in base 2, build a `PolyRingElem` from it.
fn binary_decomp_to_poly(mut a: u64) -> PolyRingElem {
    let mut result = PolyRingElem::zero();

    for i in 0..DEGREE_SIZE {
        if a & 1 != 0 {
            result.element[i] = 1.into();
        } else {
            result.element[i] = 0.into();
        }
        a >>= 1;
    }

    result
}

pub fn expand_witness(statement: &SimpleStatement, witness: &Witness) -> Witness {
    let r = witness.r;
    let expanded_r = r + 1;
    let mut total_norm_square: u128 = witness.norm_square.iter().sum::<u128>();

    let mut dim = witness.dim.clone();
    let mut norm_square = witness.norm_square.clone();
    let mut vectors = witness.vectors.clone();

    // build last vector: k PolyRingElem
    let mut last_vec = PolyVec::new();
    for i in 0..r {
        if statement.squared_norm_bound[i] != 0 {
            last_vec.0.push(binary_decomp_to_poly(
                (statement.squared_norm_bound[i] - witness.norm_square[i]) as u64,
            ));
        }
    }

    let last_norm_square = last_vec.norm_square();
    norm_square.push(last_norm_square);
    total_norm_square += last_norm_square;
    total_norm_square *= (*SLACK * *SLACK) as u128;

    dim.push(last_vec.0.len());
    vectors.push(last_vec);

    for i in 0..expanded_r {
        if i == r || statement.squared_norm_bound[i] == 0 {
            if total_norm_square + (total_norm_square * dim[i] as u128 * DEGREE as u128).isqrt()
                > 1 << LOG_PRIME
            {
                panic!("expand witness: Total witness norm too big for binary proof.");
            }
        } else if total_norm_square > (1 << (LOG_PRIME - 1)) - OFFSET as u128 {
            panic!("expand witness: Total witness norm too big for exact l2 proof.")
        }
    }

    Witness {
        r: expanded_r,
        dim,
        norm_square,
        vectors,
    }
}

/// Build the polynomial `sum(i, -2^(DEGREE - i) X^i)`.
pub fn build_gadget() -> PolyRingElem {
    let mut result: Vec<BaseRingElem> = Vec::with_capacity(DEGREE_SIZE);
    let max_idx = (LOG_PRIME as usize) - 1;
    for i in 0..max_idx {
        result[i] = -BaseRingElem::from(1 << (DEGREE - i as u64));
    }
    for i in max_idx..DEGREE_SIZE {
        result[i] = BaseRingElem::zero();
    }

    PolyRingElem { element: result }
}

pub fn simple_commit(
    output_stat: &mut Statement,
    output_wit: &mut Witness,
    proof: &mut Proof,
    commit: &SimpleCommitment,
    input_stat: &SimpleStatement,
    expanded_wit: &Witness,
) -> Witness {
    let r = expanded_wit.r;
    let dim = &expanded_wit.dim;
    let com_params = &proof.commit_params;

    let s1 = proof.chunks[r - 1];
    let s2 = s1 + proof.chunks[r];
    let s3 = s2 + proof.chunks[2 * r];

    let t2_offset = s2 * com_params.uniform_length * com_params.commit_rank_1;
    let quad_offset = s3 * com_params.uniform_length * com_params.commit_rank_1;

    let gadget = build_gadget();
    let packed_wit = Witness {
        r: output_stat.r,
        dim: vec![output_stat.dim; output_stat.r],
        norm_square: vec![0_u128; r],
        vectors: vec![PolyVec::new(); r],
    };

    let mut first_half: Vec<PolyVec> = expanded_wit.vectors.clone();
    let mut mid: PolyVec = PolyVec::zero(r);
    let mut second_half: Vec<PolyVec> = Vec::with_capacity(r);
    // TODO: build packed wit
    for i in 0..(expanded_wit.r - 1) {
        let mut current_polyvec = first_half[i].clone();
        if input_stat.squared_norm_bound[i] != 0 {
            current_polyvec.invert_x();
        } else {
            // FIXME: implement `flip` (ie polxvec_flip)
            current_polyvec.flip();
        }
        second_half.push(current_polyvec);
    }

    packed_wit
}

pub fn simple_prove(input_stat: &SimpleStatement, input_wit: &Witness, tail: bool) {
    let expanded_wit = expand_witness(input_stat, input_wit);
    let mut proof = Proof::new(&expanded_wit, 2, tail);

    let mut commitments = SimpleCommitment::new(&proof.commit_params, &expanded_wit);
    let mut output_stat =
        Statement::new(&proof, &input_stat.hash, Some(input_stat.commit_key.seed));
    let mut output_wit = Witness::new(&output_stat);

    let s1 = proof.chunks[input_stat.r];

    let now = Instant::now();
    let packed_wit = simple_commit(
        &mut output_stat,
        &mut output_wit,
        &mut proof,
        &commitments,
        input_stat,
        &expanded_wit,
    );
    println!(
        "(dachshund) Committed ({} sec), hash: {:?}",
        now.elapsed().as_secs_f32(),
        output_stat.hash
    );

    let now = Instant::now();

    let jl_matrices = project(&mut output_stat, &mut proof, &expanded_wit);

    println!(
        "Projected ({} sec), hash: {:?}",
        now.elapsed().as_secs_f32(),
        output_stat.hash
    );

    for i in 0..U128_LEN {
        // REVIEW: maybe wrong rank r (should be s1 ?)
        let mut constraint = collaps_jl_matrices(&mut output_stat, &proof, &jl_matrices);
        let zero_idx = s1 * output_stat.dim;
        constraint.linear_part.0[zero_idx..].fill(PolyRingElem::zero());
        // TODO: collaps sparse constraint
        //       simple collaps
        //       gen_lifting_polynomials
    }

    todo!()
}

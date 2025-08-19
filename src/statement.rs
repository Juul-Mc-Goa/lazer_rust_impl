use crate::{
    Seed,
    commit::{CommitKey, CommitParams, Commitments},
    constraint::Constraint,
    proof::Proof,
    ring::PolyRingElem,
    utils::next_2_power,
};

#[allow(dead_code)]
#[derive(Debug)]
pub struct Statement {
    /// total number of vectors  (amortized)
    pub r: usize,
    /// dimension of amortized vectors
    pub dim: usize,
    /// dimension of inner commitments and garbage commitments
    pub dim_inner: usize,
    pub tail: bool,
    pub commit_params: CommitParams,
    pub commitments: Commitments,
    pub challenges: Vec<PolyRingElem>,
    pub constraint: Constraint,
    pub squared_norm_bound: u128,
    pub hash: [u8; 16],
    pub commit_key: CommitKey,
}

/// Computes `len.div_ceil(deg2) * deg2` where `deg2 = next_2_power(deg)`.
#[allow(dead_code)]
pub fn extlen(len: usize, deg: usize) -> usize {
    if deg == 1 {
        len
    } else {
        let mask = (next_2_power(deg as u64) - 1) as usize;
        (len + mask) & !mask
    }
}

#[allow(dead_code)]
impl Statement {
    /// Create a new `Statement` and the commitment key.
    pub fn new(proof: &Proof, hash: &[u8; 16], seed: Option<Seed>) -> Self {
        let r: usize = proof.chunks.iter().sum();

        let mut max_dim: usize = 0;
        let mut dim_acc: usize = 0;

        for i in 0..proof.r {
            dim_acc += proof.dim[i];

            if proof.chunks[i] != 0 {
                max_dim = max_dim.max(dim_acc.div_ceil(proof.chunks[i]));
                dim_acc = 0;
            }
        }

        let com_params = proof.commit_params.clone();
        let CommitParams {
            z_base: _,
            z_length: _,
            uniform_base: _,
            uniform_length: unif_len,
            quadratic_base: _,
            quadratic_length: quad_len,
            commit_rank_1: com_rank1,
            ..
        } = com_params;

        // inner commitments dimmension
        let mut dim_inner = r * unif_len * com_rank1;
        // garbage dimmension
        dim_inner += (unif_len + quad_len) * r * (r + 1) / 2;

        let seed = seed.unwrap_or_default();

        Self {
            r,
            dim: max_dim,
            dim_inner,
            tail: proof.tail,
            commit_params: com_params,
            commitments: Commitments::new(proof.tail),
            challenges: Vec::new(),
            constraint: Constraint::new(),
            squared_norm_bound: 0,
            hash: *hash,
            commit_key: CommitKey::new(proof.tail, r, max_dim, &com_params, seed),
        }
    }

    pub fn as_mut_parts(
        &mut self,
    ) -> (
        &mut usize,
        &mut usize,
        &mut usize,
        &mut bool,
        &mut CommitParams,
        &mut Commitments,
        &mut Vec<PolyRingElem>,
        &mut Constraint,
        &mut u128,
        &mut [u8; 16],
    ) {
        (
            &mut self.r,
            &mut self.dim,
            &mut self.dim_inner,
            &mut self.tail,
            &mut self.commit_params,
            &mut self.commitments,
            &mut self.challenges,
            &mut self.constraint,
            &mut self.squared_norm_bound,
            &mut self.hash,
        )
    }

    pub fn print(&self) {
        println!("r: {}", self.r);
        println!("dim: {}", self.dim);
        println!("dim_inner: {}", self.dim_inner);
        println!("tail: {}", self.tail);

        println!("commit_params:");
        println!("  z_base: {}", self.commit_params.z_base);
        println!("  z_length: {}", self.commit_params.z_length);
        println!("  uniform_base: {}", self.commit_params.uniform_base);
        println!("  uniform_length: {}", self.commit_params.uniform_length);
        println!("  quad_base: {}", self.commit_params.quadratic_base);
        println!("  quad_length: {}", self.commit_params.quadratic_length);
        println!("  commit_rank_1: {}", self.commit_params.commit_rank_1);
        println!("  commit_rank_2: {}", self.commit_params.commit_rank_2);
        println!("  u1_len: {}", self.commit_params.u1_len);
        println!("  u2_len: {}", self.commit_params.u2_len);

        println!("commitments: elided");
        println!("challenges:");
        for challenge in self.challenges.iter() {
            println!("  {:?}", challenge);
        }
        println!(
            "constraint: elided (non-zero quad coefs: {}, size of linear part: {})",
            self.constraint.quadratic_part.0.len(),
            self.constraint.linear_part.0.len(),
        );
        println!(
            "squared_norm_bound: {}, (ie around 2^{})",
            self.squared_norm_bound,
            self.squared_norm_bound.ilog2()
        );
        println!("hash: {:?}", self.hash);
        println!("commit key: \n{:?}", self.commit_key);
    }
}

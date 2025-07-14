use crate::{
    matrices::PolyMatrix,
    ring::{BaseRingElem, PolyRingElem},
};

#[allow(dead_code)]
pub struct Witness {
    /// Number of vectors (ie length of `self.vectors`).
    pub r: usize,
    /// Dimension of each vector.
    pub dim: usize,
    /// The list of vectors.
    pub vectors: Vec<Vec<PolyRingElem>>,
}

#[allow(dead_code)]
pub struct Constraint {
    pub degree: usize,
    pub quadratic_part: PolyMatrix,
    pub linear_part: Vec<PolyRingElem>,
    pub constant: PolyRingElem,
}

#[allow(dead_code)]
pub struct CommitParams {
    pub z_base: usize,
    pub z_length: usize,
    pub uniform_base: usize,
    pub uniform_length: usize,
    pub quadratic_base: usize,
    pub quadratic_length: usize,
    pub commit_rank_1: usize,
    pub commit_rank_2: usize,
    pub u1_len: usize,
    pub u2_len: usize,
}

#[allow(dead_code)]
pub struct OuterCommit {
    u1: Vec<PolyRingElem>,
    u2: Vec<PolyRingElem>,
}

#[allow(dead_code)]
pub struct Proof {
    pub r: usize,
    pub dim: Vec<usize>,
    pub wit_length: Vec<usize>,
    pub tail: bool,
    pub commit_params: CommitParams,
    pub outer_commit_1: Vec<PolyRingElem>,
    pub jl_nonce: usize,
    pub projection: [BaseRingElem; 256],
    pub norm_square: u64,
}

#[allow(dead_code)]
pub struct Statement {
    pub outer_commit: OuterCommit,
    pub challenges: Vec<PolyRingElem>,
    pub constraint: Constraint,
    pub norm_bound: u64,
    pub hash: [u8; 16],
}

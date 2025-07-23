use crate::{
    commit::{CommitKey, CommitParams, Commitments},
    constraint::Constraint,
    proof::Proof,
    ring::PolyRingElem,
    utils::next_2_power,
};

#[allow(dead_code)]
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
    pub squared_norm_bound: u64,
    pub hash: [u8; 16],
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
    pub fn new(proof: &Proof, hash: &[u8; 16]) -> (Self, CommitKey) {
        let r: usize = proof.wit_length.iter().sum();

        let mut max_dim: usize = 0;
        let mut dim_acc: usize = 0;

        for i in 0..proof.r {
            dim_acc += proof.dim[i];

            if proof.wit_length[i] != 0 {
                max_dim = max_dim.max(dim_acc.div_ceil(proof.wit_length[i]));
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
            // commit_rank_2: com_rank2,
            ..
        } = com_params;

        // inner commitments dimmension
        let mut dim_inner = r * unif_len * com_rank1;
        // garbage dimmension
        dim_inner += (unif_len + quad_len) * r * (r + 1) / 2;

        // NOTE: legacy code ahead
        //
        // len is a multiple of next_2_power(com_rank2)
        // len is bigger than unif_len * com_rank1
        // let mut len = r * extlen(unif_len * com_rank1, com_rank2);
        // len += quad_len * r * (r + 1) / 2;
        // len = len.max(unif_len * r * (r + 1) / 2).max(max_dim);

        (
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
            },
            CommitKey::new(proof.tail, r, max_dim, com_params),
        )
    }
}

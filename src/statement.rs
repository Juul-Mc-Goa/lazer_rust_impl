use crate::{
    commit::{CommitParams, OuterCommit},
    constraint::Constraint,
    proof::Proof,
    ring::PolyRingElem,
    witness::next_2_power,
};

#[allow(dead_code)]
pub struct Statement {
    pub r: usize,
    pub dim: usize,
    pub dim_inner: usize,
    pub tail: bool,
    pub commit_params: CommitParams,
    pub outer_commit: OuterCommit,
    pub challenges: Vec<PolyRingElem>,
    pub constraint: Constraint,
    pub squared_norm_bound: u64,
    pub hash: [u8; 16],
}

pub fn extlen(len: usize, deg: usize) -> usize {
    if deg == 1 {
        len
    } else {
        let mask = (next_2_power(deg as u64) - 1) as usize;
        (len + mask) & !mask
    }
}

impl Statement {
    /// Create a new `Statement`, and return the minimum length of the commitment key.
    pub fn new(proof: &Proof, hash: &[u8; 16]) -> (Self, usize) {
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
        // inner commitments dimmension
        let mut dim_inner = r * com_params.uniform_length * com_params.commit_rank_1;
        // garbage dimmension
        dim_inner += (com_params.uniform_length + com_params.quadratic_length) * r * (r + 1) / 2;

        let mut len = r * extlen(
            com_params.uniform_length * com_params.commit_rank_1,
            com_params.commit_rank_2,
        );
        len += com_params.quadratic_length * r * (r + 1) / 2;
        len = len.max(com_params.uniform_length * r * (r + 1) / 2);
        len = len.max(max_dim);

        let outer_commit = OuterCommit {
            u1: Vec::new(),
            u2: Vec::new(),
        };

        (
            Self {
                r,
                dim: max_dim,
                dim_inner,
                tail: proof.tail,
                commit_params: com_params,
                outer_commit,
                challenges: Vec::new(),
                constraint: Constraint::new(),
                squared_norm_bound: 0,
                hash: *hash,
            },
            len,
        )
    }
}

use crate::{proof::Proof, ring::PolyRingElem, statement::Statement, witness::Witness};

#[allow(dead_code)]
#[derive(Clone)]
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
    pub u1: Vec<PolyRingElem>,
    pub u2: Vec<PolyRingElem>,
}

pub fn commit(
    output_stat: &mut Statement,
    output_wit: &mut Witness,
    proof: &mut Proof,
    input_wit: &Witness,
) -> Vec<Vec<PolyRingElem>> {
    let r = input_wit.r;
    let dim = &input_wit.dim;
    let com_params = &output_stat.commit_params;

    // v is the concatenation of (commitment to iwt.vectors) and (garbage)
    let (_, v) = output_wit.vectors.split_at(com_params.z_length);
    let (commitment, garbage) =
        v.split_at(output_stat.r * com_params.uniform_length * com_params.commit_rank_1);

    output_stat.outer_commit.u1 = vec![PolyRingElem::zero(); com_params.commit_rank_1];

    // buf: sx + jl_matrices
    // sx: r*n polynomials
    // jl_matrices r*n * (256 * DEGREE / 8) bytes

    // FIXME
    Vec::new()
}

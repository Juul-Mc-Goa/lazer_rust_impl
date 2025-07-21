use crate::{matrices::SparsePolyMatrix, ring::PolyRingElem};

/// A degree 2 polynomial in the `r` witness vectors s_1, ..., s_r in R_q^n:
///
/// sum(a_{i,j} <s_i, s_j>) + sum(<b_i, s_i>) + c
///
/// Here:
/// * a_{i,j} is in R_q
/// * b_i is in R_q^n
/// * c is in R_q
#[allow(dead_code)]
pub struct Constraint {
    pub degree: usize,
    pub quadratic_part: SparsePolyMatrix,
    pub linear_part: Vec<PolyRingElem>,
    pub constant: PolyRingElem,
}

impl Constraint {
    pub fn new_raw(degree: usize) -> Self {
        Self {
            degree,
            quadratic_part: SparsePolyMatrix(Vec::new()),
            linear_part: Vec::new(),
            constant: PolyRingElem::zero(),
        }
    }
    pub fn new() -> Self {
        Self::new_raw(1)
    }
}

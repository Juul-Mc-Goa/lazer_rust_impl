use crate::{matrices::SparsePolyMatrix, ring::PolyRingElem};

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

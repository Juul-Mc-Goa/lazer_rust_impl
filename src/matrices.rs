use crate::ring::{BaseRingElem, PolyRingElem};

pub struct PolyMatrix(pub Vec<Vec<PolyRingElem>>);
#[allow(dead_code)]
pub struct BaseMatrix(pub Vec<Vec<BaseRingElem>>);

/// Sparse matrix with polynomial coefficients: a vector of
/// `(row_index, column_index, coef)`.
pub struct SparsePolyMatrix(pub Vec<(usize, usize, PolyRingElem)>);

impl From<PolyMatrix> for BaseMatrix {
    fn from(_value: PolyMatrix) -> Self {
        todo!()
    }
}

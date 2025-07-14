use crate::ring::{BaseRingElem, PolyRingElem};

pub struct PolyMatrix(pub Vec<Vec<PolyRingElem>>);
#[allow(dead_code)]
pub struct BaseMatrix(pub Vec<Vec<BaseRingElem>>);

impl From<PolyMatrix> for BaseMatrix {
    fn from(_value: PolyMatrix) -> Self {
        todo!()
    }
}

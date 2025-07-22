use crate::ring::{BaseRingElem, PolyRingElem};

use rand_chacha::ChaCha8Rng;

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

impl PolyMatrix {
    /// Generate a random matrix of the given size, using the given RNG.
    pub fn random(rng: &mut ChaCha8Rng, rows: usize, cols: usize) -> Self {
        let mut result = vec![vec![PolyRingElem::zero(); cols]; rows];

        result
            .iter_mut()
            .for_each(|row| row.fill_with(|| PolyRingElem::random(rng)));

        Self(result)
    }

    /// Compute `output = output + self * input`.
    pub fn add_apply(&self, output: &mut [PolyRingElem], input: &[PolyRingElem]) {
        if input.len() != self.0[0].len() {
            panic!(
                "Applying matrix to incompatible vector: matrix rows = {}, vector length = {}",
                self.0[0].len(),
                input.len()
            );
        }
        if output.len() != self.0.len() {
            panic!(
                "Storing matrix to incompatible vector: matrix colums = {}, vector length = {}",
                self.0.len(),
                output.len()
            );
        }

        for (i, line) in self.0.iter().enumerate() {
            for (coef, coord) in line.iter().zip(input.iter()) {
                output[i] += coef * coord;
            }
        }
    }

    pub fn apply(&self, input: &[PolyRingElem]) -> Vec<PolyRingElem> {
        let mut output = vec![PolyRingElem::zero(); self.0.len()];
        self.add_apply(&mut output, input);

        output
    }
}

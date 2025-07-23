use crate::linear_algebra::PolyVec;
use crate::statement::Statement;

#[allow(dead_code)]
pub struct Witness {
    /// Number of vectors (ie length of `self.vectors`).
    pub r: usize,
    /// Dimension of each vector.
    pub dim: Vec<usize>,
    /// Squared norm of each vector
    pub norm_square: Vec<u128>,
    /// The list of vectors.
    pub vectors: Vec<PolyVec>,
}

#[allow(dead_code)]
impl Witness {
    /// Create a witness with correct capacities.
    pub fn new_raw(r: usize, dim: Vec<usize>) -> Self {
        let len = dim.iter().sum();
        Self {
            r,
            dim,
            norm_square: Vec::with_capacity(r),
            vectors: Vec::with_capacity(len),
        }
    }

    /// Initialize a witness for a given `Statement`.
    pub fn new(statement: &Statement) -> Self {
        let mut r = statement.r;
        let mut dim = vec![statement.dim; statement.commit_params.z_length];

        if !statement.tail {
            r += 1;
            dim.push(statement.dim_inner);
        }

        Self::new_raw(r, dim)
    }
}

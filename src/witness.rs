use crate::constraint::Constraint;
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

    /// Compute linear garbage terms: a vector of size `r(r+1)/2` built from
    /// `self` and `constraint.linear_part`.
    pub fn linear_garbage(&self, constraint: &Constraint) -> PolyVec {
        let mut result = PolyVec(Vec::with_capacity(self.r * (self.r + 1) / 2));

        for i in 0..self.r {
            for j in 0..=i {
                // result[i(i+1)/2 + j] = (<lin_part[i], wit[j]> + <lin_part[j], wit[i]>) / 2
                let part_1 = constraint.linear_part[i].scalar_prod(&self.vectors[j]);
                let part_2 = constraint.linear_part[j].scalar_prod(&self.vectors[i]);
                result.0.push((part_1 + part_2).halve());
            }
        }

        result
    }

    /// Compute linear garbage terms: a vector of size `r(r+1)/2` containing all
    /// scalar products `< self.vectors[i], self.vectors[j] >` for `0 <= j <= i
    /// < r`.
    pub fn quadratic_garbage(&self) -> PolyVec {
        let mut result = PolyVec(Vec::with_capacity(self.r * (self.r + 1) / 2));

        for i in 0..self.r {
            for j in 0..=i {
                result.0.push(self.vectors[i].scalar_prod(&self.vectors[j]));
            }
        }

        result
    }
}

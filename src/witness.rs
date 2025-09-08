use crate::{linear_algebra::PolyVec, statement::Statement};

#[allow(dead_code)]
#[derive(Clone, Debug)]
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
        let mut r = statement.commit_params.z_length;
        let mut dim = vec![statement.dim; statement.commit_params.z_length];

        if !statement.tail {
            r += 1;
            dim.push(statement.dim_inner);
        }

        Self::new_raw(r, dim)
    }

    pub fn print(&self) {
        println!("  r: {}", self.r);
        println!("  dim: {:?}", self.dim);
        // println!("  vectors: {:?}", self.vectors);
    }

    /// Decompose a witness `(s_1, ..., s_r)` in `base`: `s_i = sum(k, (base ^
    /// k) s_ik)`. Return a new witness `(s_{1, 1}, ..., s_{r, 1}, ..., s_{1,
    /// len}, ..., s_{r, len})`.
    pub fn decomp(self, base: usize, len: usize) -> Self {
        let mut chunks: Vec<Vec<PolyVec>> = vec![Vec::with_capacity(self.r); len];
        let mut norm_chunks: Vec<Vec<u128>> = vec![Vec::with_capacity(self.r); len];

        for v in self.vectors {
            for (i, small_v) in v.decomp(base, len).into_iter().enumerate() {
                norm_chunks[i].push(small_v.norm_square());
                chunks[i].push(small_v);
            }
        }

        let mut vectors: Vec<PolyVec> = Vec::with_capacity(self.r * len);
        let mut norm_square: Vec<u128> = Vec::with_capacity(self.r * len);

        chunks
            .into_iter()
            .for_each(|mut chunk| vectors.append(&mut chunk));

        norm_chunks
            .into_iter()
            .for_each(|mut chunk| norm_square.append(&mut chunk));

        Self {
            r: len * self.r,
            dim: self.dim,
            norm_square,
            vectors,
        }
    }

    /// Append `other` witness to `self`.
    pub fn concat(&mut self, other: &mut Self) {
        if self.dim != other.dim {
            panic!(
                "witness concatenation: dimensions don't match (self: {:?}, other: {:?})",
                self.dim, other.dim
            );
        }

        self.r += other.r;
        self.norm_square.append(&mut other.norm_square);
        self.vectors.append(&mut other.vectors);
    }

    /// Post-process `self` after one round of reduction: this means
    /// splitting the last vector and updating the other fields accordingly.
    pub fn post_process(&mut self) {
        let dim = self.vectors[0].0.len();
        let mut tgh = self.vectors.pop().unwrap().into_chunks(dim);
        self.vectors.append(&mut tgh);

        self.r = self.vectors.len();
        let r = self.r;
        self.dim = vec![dim; r];

        self.norm_square = self.vectors.iter().map(|v| v.norm_square()).collect();
    }
}

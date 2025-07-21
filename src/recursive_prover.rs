use crate::{constraint::Constraint, matrices::PolyMatrix, ring::PolyRingElem};

#[allow(dead_code)]
pub struct RecursiveProver {
    /// Dimension of each witness vector
    pub n: usize,
    /// Number of witnesses
    pub r: usize,
    /// The witnesses: a (r,n) matrix, coefficients in `PolyRing`
    pub witnesses: PolyMatrix,
    /// List of degree 2 polynomials over `R_q` that should equal zero when
    /// evaluated on the witnesses
    pub zero_constraints: Vec<Constraint>,
    /// List of degree 2 polynomials over `R_q` that should have zero constant
    /// coefficients when evaluated on the witnesses
    pub coef_constraints: Vec<Constraint>,
    /// The l2 norm bound the witnesses verify
    pub norm_bound: u64,
    /// The matrix used for committing to the witnesses
    pub s_commit_mat_1: PolyMatrix,
    /// First commit
    pub s_commit_1: PolyMatrix,
    /// The matrix used for committing to `s_commit_1`
    pub s_commit_mat_2: PolyMatrix,
    /// Second commit
    pub s_commit_2: Vec<PolyRingElem>,
    /// Matrix to commit to quadratic garbage
    pub g_commit_mat: PolyMatrix,
    /// Matrix to commit to quadratic garbage
    pub h_commit_mat: PolyMatrix,
    /// Aggregate of secrets s_i
    pub z: Vec<PolyRingElem>,
    /// Commit to the quadratic garbages (uses `g_commit_mat` and `h_commit_mat`)
    pub quad_commit: Vec<PolyRingElem>,
}

#[allow(dead_code)]
impl RecursiveProver {
    pub fn new(
        witnesses: PolyMatrix,
        s_commit_mat: (PolyMatrix, PolyMatrix),
        quadratic_commit_mat: (PolyMatrix, PolyMatrix),
    ) -> Self {
        let n = witnesses.0[0].len();
        let r = witnesses.0.len();
        Self {
            n,
            r,
            witnesses,
            zero_constraints: Vec::new(),
            coef_constraints: Vec::new(),
            norm_bound: 0,
            s_commit_mat_1: s_commit_mat.0,
            s_commit_1: PolyMatrix(Vec::new()),
            s_commit_mat_2: s_commit_mat.1,
            s_commit_2: Vec::new(),
            g_commit_mat: quadratic_commit_mat.0,
            h_commit_mat: quadratic_commit_mat.1,
            z: Vec::new(),
            quad_commit: Vec::new(),
        }
    }

    pub fn set_norm_constraint(&mut self, beta: u64) {
        self.norm_bound = beta;
    }

    pub fn add_zero_constraint(&mut self, constraint: Constraint) {
        self.zero_constraints.push(constraint);
    }

    pub fn project(&mut self) {
        todo!()
    }
}

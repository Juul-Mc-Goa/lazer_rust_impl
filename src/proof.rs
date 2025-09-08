use crate::{
    commit::{CommitParams, Commitments},
    constants::{CHALLENGE_NORM, DEGREE, LOG_DELTA, LOG_PRIME, SLACK, TAU1, TAU2},
    linear_algebra::PolyVec,
    ring::{BaseRingElem, PolyRingElem},
    witness::Witness,
};

#[allow(dead_code)]
pub struct Proof {
    /// Number of vectors inside the witness.
    pub r: usize,
    /// Witness dimensions.
    pub dim: Vec<usize>,
    /// Witness dimension after splitting.
    pub split_dim: usize,
    /// Decomposition parts.
    pub chunks: Vec<usize>,
    /// `true` if this proof is the last one of a composite proof.
    pub tail: bool,
    /// Commitment parameters.
    pub commit_params: CommitParams,
    /// Commitments vectors.
    pub commitments: Commitments,
    /// Nonce used for generating the JL matrix.
    pub jl_nonce: u128,
    /// The vector obtained by applying the JL matrix to the witness.
    pub projection: [BaseRingElem; 256],
    /// A_p -> R_p lifting polynomials
    pub lifting_poly: Vec<PolyRingElem>,
    pub norm_square: u128,
}

/// Check that `norm` is small enough for the current global parameters and a
/// lattice of rank `rank`.
pub fn sis_secure(rank: usize, norm: f64) -> bool {
    let rank = rank as u64;
    let mut maxlog: f64 = 2.0 * (*LOG_DELTA * (LOG_PRIME * DEGREE * rank) as f64).sqrt();
    maxlog = maxlog.min(LOG_PRIME as f64);

    norm.log2() < maxlog
}

/// Split `z`, compute its decomposition base and length.
///
/// Return
/// - `split_dim`: the new dimension after splitting
/// - `z_length`: decomposition length
/// - `z_base`: decomposition base
/// - `r`: new rank after splitting
/// - `var_z`: variance of each coordinate in `z`
fn z_decompose(
    k: usize,
    is_tail: bool,
    chunks: &Vec<usize>,
    new_chunks: &mut Vec<usize>,
    norm_square: &Vec<u128>,
    max_dim: usize,
) -> (usize, usize, usize, usize, u128) {
    let split_dim = max_dim.div_ceil(k);

    // decompose witness as vectors of size split_dim
    *new_chunks = chunks.iter().map(|l| l.div_ceil(split_dim)).collect();

    let new_r: usize = new_chunks.iter().sum();

    let mut var_z: u128 = norm_square.iter().sum();
    // average square of each coefficient:
    var_z /= split_dim as u128 * DEGREE as u128;
    // times the variance of the challenges:
    let var_challenges = TAU1 + 4 * TAU2;
    var_z *= var_challenges;

    let mut decompose: bool = {
        let challenge_norm = CHALLENGE_NORM as f64;
        let var_challenges = var_challenges as f64;
        let var_z = var_z as f64;
        let split_dim = split_dim as f64;
        let degree = DEGREE as f64;

        !is_tail
            && !sis_secure(
                13,
                6.0 * challenge_norm
                    * *SLACK
                    * (2.0 * var_challenges * var_z * split_dim * degree).sqrt(),
            )
    };

    // decompose if average squared coefficient is greater than JL_MAX_NORM
    decompose |= DEGREE as u128 * var_z > (1 << 28);

    let twelve_log2 = |i| (12_f64 * i as f64).log2();

    if decompose {
        // z_base =  log((12 * varz)^1/4)
        let z_base = (twelve_log2(var_z) / 4.0).ceil() as usize;
        (split_dim, 2, z_base, new_r, var_z)
    } else {
        // z_base = log((12 * varz)^1/2)
        let z_base = (twelve_log2(var_z) / 2.0).ceil() as usize;
        (split_dim, 1, z_base, new_r, var_z)
    }
}

/// Compute `quadratic_length` and `var_g` (the variance of each coordinate in
/// `quad_garbage`).
fn quad_length_varg(
    is_tail: bool,
    r: usize,
    quadratic: u8,
    norm_square: &Vec<u128>,
    dim: &Vec<usize>,
    z_base: usize,
    split_dim: usize,
    chunks: &Vec<usize>,
) -> (usize, u128) {
    // this will hold the variance of ONE coefficient of the `g_ij`s.
    let mut var_g: u128 = 0;

    if quadratic == 0 {
        return (0, var_g);
    } else if is_tail {
        return (1, var_g);
    } else {
        let mut t: usize = 0;
        let mut u: usize = 0;

        for i in 0..r {
            // Variance of witness coordinates (over A_p)
            let var_wit: f64 = norm_square[i] as f64 / (dim[i] as f64 * DEGREE as f64);
            // quadratic garbage: compute square of `var_wit`
            let var_wit_sq: usize = (var_wit * var_wit) as usize;
            let mut j: usize = dim[i];

            // the vector g is split into many vectors of dimension `split_dim`
            while j >= split_dim - u {
                j -= split_dim - u;
                t += var_wit_sq * (split_dim - u);
                var_g = var_g.max(t as u128);
                t = 0;
                u = 0;
            }

            t += var_wit_sq * j;
            u += j;

            if chunks[i] != 0 {
                var_g = var_g.max(t as u128);
                t = 0;
                u = 0;
            }
        }

        var_g *= 2 * DEGREE as u128;
        let quadratic_length = {
            let log_12 = 12_f64.log2();
            let log_var_g = (var_g as f64).log2();
            let z_base_f = z_base as f64;

            ((log_12 + log_var_g) / (2.0 * z_base_f)).ceil() as usize
        };

        return (quadratic_length.max(1_usize), var_g);
    }
}

/// Try each possibility for `commit_rank_1`, return it and the total norm
/// if this possibility is correct (i.e. the total norm is SIS-secure).
fn commit_rank_1_total_norm(
    is_tail: bool,
    quadratic: u8,
    varz: u128,
    varg: u128,
    split_dim: usize,
    z_base: usize,
    z_length: usize,
    uniform_base: usize,
    uniform_length: usize,
    quadratic_base: usize,
    quadratic_length: usize,
    new_r: usize,
) -> (usize, u128) {
    let quad_rank = new_r * (new_r + 1) / 2;
    let (unif_base, unif_len) = (uniform_base as u128, uniform_length as u128);
    let (z_base, z_len) = (z_base as u128, z_length as u128);
    let (quad_base, quad_len) = (quadratic_base as u128, quadratic_length as u128);

    let mut commit_rank_1: usize = 0;
    let mut total_norm_square: u128 = 0;

    // variance of a vector with `length - 1` coordinates, each uniform in
    // 0..2^base
    let var_decomp = |base: u128, length: u128| ((length - 1) << 2 * base) / 12;
    // variance of the highest digit
    let var_highest = |base: u128, length: u128, var: u128| var >> (base * (length - 1));

    let var_z_part =
        (var_decomp(z_base, z_len) + var_highest(z_base, z_len, varz)) * split_dim as u128;

    while commit_rank_1 <= 32 {
        commit_rank_1 += 1;

        total_norm_square = var_z_part * split_dim as u128;

        if !is_tail {
            let t_rank = new_r * commit_rank_1;
            let var_unif_low = var_decomp(unif_base, unif_len);
            let var_unif_high = (1_u128 << 2 * LOG_PRIME as u128)
                .unbounded_shr(2 * (unif_base * (unif_len - 1)) as u32)
                / 12;

            total_norm_square += (var_unif_low + var_unif_high) * (t_rank + quad_rank) as u128;

            if quadratic != 0 {
                let var_quad_low = var_decomp(quad_base, quad_len);
                let var_quad_high = var_highest(quad_base, quad_len, varg);

                total_norm_square += (var_quad_low + var_quad_high) * quad_rank as u128;
            }
        }

        total_norm_square *= DEGREE as u128;

        // if it's sis secure, then commit_rank_1 is good: break
        if sis_secure(
            commit_rank_1,
            6.0 * CHALLENGE_NORM as f64
                * *SLACK
                * (1 << (z_len - 1) * z_base) as f64
                * (total_norm_square as f64).sqrt(),
        ) {
            break;
        }
    }

    (commit_rank_1, total_norm_square)
}

/// Compute `commit_rank_2, u1_len, u2_len`.
fn commit_2_u1_u2(
    is_tail: bool,
    quadratic: u8,
    new_r: usize,
    total_norm: f64,
    varz: u128,
    split_dim: usize,
    commit_rank_1: usize,
    uniform_length: usize,
    quadratic_length: usize,
) -> (bool, usize, usize, usize) {
    let mut commit_rank_2: usize = 0;
    let u1_len: usize;
    let u2_len: usize;

    if !is_tail {
        // compute commit_rank_2
        while commit_rank_2 <= 32 {
            commit_rank_2 += 1;
            if sis_secure(commit_rank_2, 2.0 * *SLACK * total_norm) {
                break;
            }
        }

        // compute outer commitment length
        u1_len = commit_rank_2;
        u2_len = commit_rank_2;

        // check the commit_ranks are OK
        if commit_rank_1 <= 32
            && commit_rank_2 <= 32
            // check the total dimension of the concatenated commitment vector
            // (t || g || h)
            && uniform_length * new_r * commit_rank_1
            + (uniform_length + quadratic_length) * new_r * (new_r + 1) / 2
            <= (1.1 * split_dim as f64) as usize
        {
            // every parameter is correct: return
            return (true, commit_rank_2, u1_len, u2_len);
        }
    } else {
        commit_rank_2 = 0;
        // compute outer commitment lengths
        u1_len = if quadratic != 0 {
            new_r * commit_rank_1 + new_r * (new_r + 1) / 2
        } else {
            new_r * commit_rank_1
        };
        u2_len = 2 * new_r - 1;

        if commit_rank_1 <= 32
            && (u1_len + u2_len) * LOG_PRIME as usize
                <= (1.1 * (split_dim as f64) * ((varz as f64).log2() / 2.0 + 2.05)) as usize
        {
            // every parameter is correct: return
            return (true, commit_rank_2, u1_len, u2_len);
        }
    }

    // didn't find a correct set of parameters
    return (false, commit_rank_2, u1_len, u2_len);
}

#[allow(dead_code)]
impl Proof {
    fn init_dim_norm_chunks(
        quadratic: u8,
        r: usize,
        witness: &Witness,
    ) -> (Vec<usize>, Vec<u128>, Vec<usize>) {
        let mut dim = vec![0_usize; r]; // dimension of each vector (before splitting)
        // vectors will be split into `dim.div_ceil(chunks[i])` vectors of rank `chunks[i]`.
        let mut chunks = vec![0_usize; r];
        let mut norm_square = vec![0_u128; r]; // squared l2 norm

        if quadratic == 2 {
            for i in 0..witness.r {
                dim[i] = witness.dim[i];
                dim[witness.r + 1 + i] = dim[i];
                norm_square[i] = witness.norm_square[i];
                norm_square[witness.r + 1 + i] = (TAU1 + 4 * TAU2) * witness.norm_square[i];
            }
            // A_p to R_p liftings
            dim[witness.r] = (LOG_PRIME as f64 / 10.0).ceil() as usize * witness.r;
            norm_square[witness.r] = (dim[witness.r] as u64 * DEGREE * (1 << 20) / 12) as u128;

            chunks[witness.r - 1] = 1;
            chunks[witness.r] = 1;
            chunks[2 * witness.r] = 1;
        } else {
            dim.copy_from_slice(&witness.dim);
            norm_square.copy_from_slice(&witness.norm_square);
            if quadratic != 0 {
                chunks.fill(1);
            } else {
                chunks.fill(0);
                chunks[r - 1] = 1;
            }
        }

        (dim, norm_square, chunks)
    }

    pub fn new(witness: &Witness, quadratic: u8, is_tail: bool) -> Self {
        // compute the number of vectors in the modified witness
        let r = if quadratic == 2 {
            2 * witness.r + 1
        } else {
            witness.r
        };

        // initialize a few vectors
        let (dim, norm_square, mut chunks) = Self::init_dim_norm_chunks(quadratic, r, witness);

        // compute global variables used in the for loop
        let mut dim_acc = 0; // current dimension of joined vectors
        let mut max_dim = 0; // max dimension of joined vectors

        for i in 0..r {
            dim_acc += dim[i];
            if chunks[i] != 0 {
                chunks[i] = dim_acc;
                max_dim = max_dim.max(dim_acc);
                dim_acc = 0;
            }
        }

        // initialize most parameters, these will be modified inside the loop
        let mut new_wit_length: Vec<usize> = Vec::new();

        let mut split_dim: usize = 0;
        let mut new_r: usize;

        let mut z_base: usize = 0;
        let mut z_length: usize = 0;
        let mut uniform_base: usize = 0;
        let mut uniform_length: usize = 0;
        let mut quadratic_base: usize = 0;
        let mut quadratic_length: usize = 0;

        let mut varz: u128;
        let mut varg: u128;

        let mut commit_rank_1: usize = 0;
        let mut commit_rank_2: usize = 0;

        let mut total_norm_square: u128 = 0;
        let mut total_norm: f64;

        let mut u1_len: usize = 0;
        let mut u2_len: usize = 0;

        for k in (1..16).rev() {
            // decompose the inner commitment (z)
            (split_dim, z_length, z_base, new_r, varz) = z_decompose(
                k,
                is_tail,
                &chunks,
                &mut new_wit_length,
                &norm_square,
                max_dim,
            );

            if !is_tail {
                uniform_length = (LOG_PRIME as usize).div_ceil(z_base);
                uniform_base = (LOG_PRIME as usize).div_ceil(uniform_length);
            } else {
                uniform_length = 1;
                uniform_base = LOG_PRIME as usize;
            };

            // decompose quadratic garbage
            quadratic_base = z_base;
            (quadratic_length, varg) = quad_length_varg(
                is_tail,
                r,
                quadratic,
                &norm_square,
                &dim,
                z_base,
                split_dim,
                &new_wit_length,
            );

            // compute commitment rank 1
            (commit_rank_1, total_norm_square) = commit_rank_1_total_norm(
                is_tail,
                quadratic,
                varz,
                varg,
                split_dim,
                z_base,
                z_length,
                uniform_base,
                uniform_length,
                quadratic_base,
                quadratic_length,
                new_r,
            );

            total_norm = (total_norm_square as f64).sqrt();

            // compute the last params: commitment rank 2, length of vectors u1 and u2
            let good_params: bool;
            (good_params, commit_rank_2, u1_len, u2_len) = commit_2_u1_u2(
                is_tail,
                quadratic,
                new_r,
                total_norm,
                varz,
                split_dim,
                commit_rank_1,
                uniform_length,
                quadratic_length,
            );

            // every parameter is correct: escape the `for k ...` loop
            if good_params {
                break;
            }
        }

        if commit_rank_1 == 33 {
            panic!("Cannot not make inner commitments secure!");
        } else if commit_rank_2 == 33 {
            panic!("Cannot not make outer commitments secure!");
        }

        // build the commitment params
        let commit_params = CommitParams {
            z_base,
            z_length,
            uniform_base,
            uniform_length,
            quadratic_base,
            quadratic_length,
            commit_rank_1,
            commit_rank_2,
            u1_len,
            u2_len,
        };

        Self {
            r,
            dim,
            split_dim,
            chunks: new_wit_length,
            tail: is_tail,
            commit_params,
            commitments: Commitments::new(is_tail),
            jl_nonce: 0,
            projection: [0.into(); 256],
            lifting_poly: vec![PolyRingElem::zero(); 256],
            norm_square: total_norm_square,
        }
    }

    /// Split `witness` according to `self.split_dim`.
    pub fn pack_witness(&self, witness: &Witness) -> Witness {
        assert_eq!(
            self.chunks.len(),
            witness.vectors.len(),
            "incompatible dimensions between proof chunks and witness size"
        );

        let mut r: usize = 0;
        let mut new_vec: Vec<PolyVec> = Vec::new();
        let mut norm_square: Vec<u128> = Vec::new();

        witness
            .vectors
            .iter()
            .zip(self.chunks.iter())
            .for_each(|(w, chunk)| {
                let split_dim = w.0.len().div_ceil(*chunk);
                w.clone()
                    .into_chunks(split_dim)
                    .into_iter()
                    .for_each(|new_w| {
                        norm_square.push(new_w.norm_square());
                        new_vec.push(new_w);
                        r += 1;
                    });
            });

        Witness {
            r,
            dim: self.dim.clone(),
            norm_square,
            vectors: new_vec,
        }
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        println!("r: {}", self.r);
        println!("dim: {:?}", self.dim);
        println!("split_dim: {:?}", self.split_dim);
        println!("chunks: {:?}", self.chunks);
        println!("tail: {:?}", self.tail);
        println!("commit_params: {:?}", self.commit_params);
        println!("jl_nonce: {:?}", self.jl_nonce);
        println!("projection: {:?}", self.projection);
        println!("norm_square: {:?}", self.norm_square);
    }
}

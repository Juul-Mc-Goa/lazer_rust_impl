use crate::{
    commit::{CommitParams, Commitments},
    constants::{CHALLENGE_NORM, DEGREE, LOG_DELTA, LOG_PRIME, SLACK, TAU1, TAU2},
    ring::{BaseRingElem, PolyRingElem},
    witness::Witness,
};

#[allow(dead_code)]
pub struct Proof {
    pub r: usize,
    /// witness dimensions
    pub dim: Vec<usize>,
    /// decomposition parts
    pub wit_length: Vec<usize>,
    /// does this proof have to be proven as is ? or reduced further ?
    pub tail: bool,
    /// Commitment parameters.
    pub commit_params: CommitParams,
    /// Commitments vectors.
    pub commitments: Commitments,
    /// Nonce used for generating the JL matrix.
    pub jl_nonce: u128,
    /// The vector obtained by applying the JL matrix to the witness.
    pub projection: [BaseRingElem; 256],
    /// A_q -> Z_q lifting polynomials
    pub lifting_poly: Vec<PolyRingElem>,
    pub norm_square: u128,
}

/// Check that `norm` is small enough for the current global parameters and a
/// lattice of rank `rank`.
pub fn sis_secure(rank: usize, norm: f64) -> bool {
    let mut maxlog: f64 = 2.0 * (*LOG_DELTA * (LOG_PRIME * DEGREE * (rank as u64)) as f64).sqrt();
    maxlog = maxlog.min(LOG_PRIME as f64);

    norm.log2() < maxlog
}

fn z_decompose(
    k: usize,
    is_tail: bool,
    wit_length: &Vec<usize>,
    new_wit_length: &mut Vec<usize>,
    norm_square: &Vec<u128>,
    max_dim: usize,
) -> (usize, usize, usize, usize, u128) {
    let decomp_dim = max_dim.div_ceil(k);

    // decompose witness as vectors of size decomp_dim
    *new_wit_length = wit_length.iter().map(|l| l.div_ceil(decomp_dim)).collect();

    let new_r: usize = new_wit_length.iter().sum();

    // compute l2 square of witness
    let mut varz: u128 = norm_square.iter().sum();
    // compute average l2 square of decomposed vectors
    // then average from R_q to A_q ?
    varz /= decomp_dim as u128 * DEGREE as u128;
    varz *= TAU1 + 4 * TAU2;

    let mut decompose: bool = !is_tail
        && !sis_secure(
            13,
            6.0 * CHALLENGE_NORM as f64
                * *SLACK
                * (2.0 * (TAU1 as f64 + 4.0 * TAU2 as f64) * DEGREE as f64).sqrt()
                * (varz as f64 * decomp_dim as f64),
        );
    // check that average squared norm is less than JL_MAX_NORMSQ ?
    decompose |= DEGREE as u128 * varz > (1 << 28);

    let twelve_log2 = |i| (12_f64 * i as f64).log2();

    if decompose {
        // z_base =  log((12 * varz)^1/4)
        (
            decomp_dim,
            2,
            (twelve_log2(varz) / 4.0).round() as usize,
            new_r,
            varz,
        )
    } else {
        // z_base = log((12 * varz)^1/2)
        (
            decomp_dim,
            1,
            (twelve_log2(varz) / 2.0).round() as usize,
            new_r,
            varz,
        )
    }
}

fn quad_length_varg(
    is_tail: bool,
    r: usize,
    quadratic: u8,
    norm_square: &Vec<u128>,
    dim: &Vec<usize>,
    z_base: usize,
    decomp_dim: usize,
    wit_length: &Vec<usize>,
) -> (usize, u128) {
    let mut varg: u128 = 0;

    if quadratic == 0 {
        return (0, varg);
    } else if is_tail {
        return (1, varg);
    } else {
        let mut t: usize = 0;
        let mut u = 0;

        // NOTE: the fuck is this ?
        for i in 0..r {
            // average of coefficients (squared)
            let vars = norm_square[i] as f64 / (dim[i] as f64 * DEGREE as f64);
            let mut j = dim[i];

            while j >= decomp_dim - u {
                j -= decomp_dim - u;
                // t is the average (squared) norm of witness[i][j..]
                t += (vars * vars) as usize * (decomp_dim - u);
                varg = varg.max(t as u128);
                t = 0;
                u = 0;
            }

            // t is the average (squared) norm of witness[i]
            t += (vars * vars) as usize * j;
            u += j;

            if wit_length[i] != 0 {
                varg = varg.max(t as u128);
                t = 0;
                u = 0;
            }
        }

        varg *= 2 * DEGREE as u128;
        let quadratic_length =
            ((12_f64.log2() + (varg as f64).log2()) / (2.0 * z_base as f64)).ceil() as usize;

        return (quadratic_length.max(1_usize), varg);
    }
}

fn commit_rank_1_total_norm(
    is_tail: bool,
    quadratic: u8,
    varz: u128,
    varg: u128,
    decomp_dim: usize,
    z_base: usize,
    z_length: usize,
    uniform_base: usize,
    uniform_length: usize,
    quadratic_base: usize,
    quadratic_length: usize,
    new_r: usize,
) -> (usize, u128) {
    let mut commit_rank_1: usize = 0;
    let mut total_norm_square: u128 = 0;

    while commit_rank_1 <= 32 {
        commit_rank_1 += 1;

        total_norm_square = ((1 << 2 * z_base as u128) / 12 * (z_length as u128 - 1)
            + varz / (1 << 2 * z_base * (z_length - 1)))
            * decomp_dim as u128;

        if !is_tail {
            // quadratic contribution: linear coefs
            let (unif_base, unif_len) = (uniform_base as u128, uniform_length as u128);
            total_norm_square += ((1 << 2 * unif_base) * (unif_len - 1)
                + (1 << 2 * (LOG_PRIME as u128 - unif_base * (unif_len - 1))))
                * (new_r * commit_rank_1 + new_r * (new_r + 1) / 2) as u128
                / 12;
        }

        if !is_tail && quadratic != 0 {
            // quadratic contribution: quadratic coefs
            let (quad_base, quad_len) = (quadratic_base as u128, quadratic_length as u128);
            total_norm_square += ((1 << 2 * quad_base) * (quad_len - 1) / 12
                + varg / (1 << 2 * quad_base * (quad_len - 1)))
                * (new_r * (new_r + 1) / 2) as u128;
        }

        total_norm_square *= DEGREE as u128;

        // if it's sis secure, then commit_rank_1 is good: break
        if sis_secure(
            commit_rank_1,
            6.0 * CHALLENGE_NORM as f64
                * *SLACK
                * (1 << (z_length - 1) * z_base) as f64
                * (total_norm_square as f64).sqrt(),
        ) {
            break;
        }
    }

    (commit_rank_1, total_norm_square)
}

fn commit_2_u1_u2(
    is_tail: bool,
    quadratic: u8,
    new_r: usize,
    total_sqrt: f64,
    varz: u128,
    decomp_dim: usize,
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
            if sis_secure(commit_rank_2, 2.0 * *SLACK * total_sqrt) {
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
            // (z + g + h)
            && uniform_length * new_r * commit_rank_1
            + (uniform_length + quadratic_length) * new_r * (new_r + 1) / 2
            <= (1.1 * decomp_dim as f64) as usize
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
                <= (1.1 * (decomp_dim as f64) * ((varz as f64).log2() / 2.0 + 2.05)) as usize
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
    pub fn new(witness: Witness, quadratic: u8, is_tail: bool) -> Self {
        // compute the number of vectors in the modified witness
        let r = if quadratic == 2 {
            2 * witness.r + 1
        } else {
            witness.r
        };

        // compute dimension of each vector, length of decomposition, squared norm
        let mut dim = vec![0_usize; r];
        let mut wit_length = vec![0_usize; r];
        let mut norm_square = vec![0_u128; r];

        if quadratic == 2 {
            for i in 0..witness.r {
                dim[i] = witness.dim[i];
                dim[witness.r + 1 + i] = dim[i];
                norm_square[i] = witness.norm_square[i];
                norm_square[witness.r + 1 + i] = (TAU1 + 4 * TAU2) * witness.norm_square[i];
            }
            // A_q to R_q liftings
            dim[witness.r] = (LOG_PRIME as f64 / 10.0).ceil() as usize * witness.r;
            norm_square[witness.r] = (dim[witness.r] as u64 * DEGREE * (1 << 20) / 12) as u128;

            wit_length[witness.r - 1] = 1;
            wit_length[witness.r] = 1;
            wit_length[2 * witness.r] = 1;
        } else {
            for i in 0..r {
                dim[i] = witness.dim[i];
                norm_square[i] = witness.norm_square[i];
                wit_length[i] = if quadratic != 0 { 1 } else { 0 };
            }

            wit_length[r - 1] = 1;
        }

        // compute global variables used in the for loop
        let mut dim_acc = 0; // current dimension of joined vectors
        let mut max_dim = 0; // max dimension of joined vectors

        for i in 0..r {
            dim_acc += dim[i];
            if wit_length[i] != 0 {
                wit_length[i] = dim_acc;
                max_dim = max_dim.max(dim_acc);
                dim_acc = 0;
            }
        }

        // initialize most parameters, these will be modified inside the loop
        let mut new_wit_length: Vec<usize> = Vec::new();

        let mut decomp_dim: usize;
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
        let mut total_sqrt: f64;

        let mut u1_len: usize = 0;
        let mut u2_len: usize = 0;

        for k in (1..16).rev() {
            // decompose the inner commitment (z)
            (decomp_dim, z_length, z_base, new_r, varz) = z_decompose(
                k,
                is_tail,
                &wit_length,
                &mut new_wit_length,
                &norm_square,
                max_dim,
            );

            (uniform_length, uniform_base) = if !is_tail {
                (
                    (LOG_PRIME as usize + 2 * z_base / 3) / z_base,
                    (LOG_PRIME as usize + z_length / 2) / z_length,
                )
            } else {
                (1, LOG_PRIME as usize)
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
                decomp_dim,
                &new_wit_length,
            );

            // compute commitment rank 1
            (commit_rank_1, total_norm_square) = commit_rank_1_total_norm(
                is_tail,
                quadratic,
                varz,
                varg,
                decomp_dim,
                z_base,
                z_length,
                uniform_base,
                uniform_length,
                quadratic_base,
                quadratic_length,
                new_r,
            );

            total_sqrt = (total_norm_square as f64).sqrt();

            // compute the last params: commitment rank 2, length of vectors u1 and u2
            let good_params: bool;
            (good_params, commit_rank_2, u1_len, u2_len) = commit_2_u1_u2(
                is_tail,
                quadratic,
                new_r,
                total_sqrt,
                varz,
                decomp_dim,
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

        // REVIEW: lifting_poly ?
        Self {
            r,
            dim,
            wit_length: new_wit_length,
            tail: is_tail,
            commit_params,
            commitments: Commitments::new(is_tail),
            jl_nonce: 0,
            projection: [0.into(); 256],
            lifting_poly: Vec::new(),
            norm_square: total_norm_square,
        }
    }
}

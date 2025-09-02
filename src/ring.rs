use num_modular::{ModularInteger, MontgomeryInt};
use rand::Rng;
use rand::SeedableRng;
use rand::distr::{Bernoulli, Distribution};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};

use crate::constants::{DEGREE, ONE_HALF_MOD_PRIME, PRIME, PRIME_BYTES_LEN, TAU1, TAU2};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// An element of `Z/pZ`, where `p = PRIME`.
#[derive(Clone, Copy, PartialEq)]
pub struct BaseRingElem {
    pub element: MontgomeryInt<u64>,
}

/// An element of `(Z/pZ)[X] / (X^d + 1)`, where `p = PRIME`, and `d = DEGREE`.
#[derive(Clone, PartialEq)]
pub struct PolyRingElem {
    pub element: Vec<BaseRingElem>,
}

/// Take residue mod `PRIME`.
impl From<u64> for BaseRingElem {
    fn from(value: u64) -> Self {
        Self {
            element: MontgomeryInt::new(value, &PRIME),
        }
    }
}

/// Trivial implementation, used when implementing `Add<T>` for `BaseRingElem`.
impl AsRef<BaseRingElem> for BaseRingElem {
    fn as_ref(&self) -> &Self {
        self
    }
}

/// Trivial implementation, used when implementing `Add<T>` for `PolyRingElem`.
impl AsRef<PolyRingElem> for PolyRingElem {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl Debug for BaseRingElem {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let residue = self.element.residue();
        let small = if 2 * residue < PRIME {
            residue as i64
        } else {
            residue as i64 - PRIME as i64
        };
        write!(f, "{small:2}")
    }
}

impl Debug for PolyRingElem {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for i in 0..DEGREE {
            if i == DEGREE - 1 {
                write!(f, "{:?}", self.element[i as usize])?;
            } else {
                write!(f, "{:?} ", self.element[i as usize])?;
            }
        }

        Ok(())
    }
}

impl From<BaseRingElem> for PolyRingElem {
    fn from(value: BaseRingElem) -> Self {
        let mut new = PolyRingElem::zero();
        new.element[0] = value;

        new
    }
}

impl BaseRingElem {
    /// Generate 0 mod `PRIME`.
    pub fn zero() -> Self {
        BaseRingElem {
            element: MontgomeryInt::new(0, &PRIME),
        }
    }
    /// Generate 1 mod `PRIME`.
    pub fn one() -> Self {
        BaseRingElem {
            element: MontgomeryInt::new(1, &PRIME),
        }
    }

    /// Return the "absolute value of `self`": the minimum of `self.element` and
    /// `PRIME - self.element`.
    pub fn abs(&self) -> u64 {
        let residue = self.element.residue();
        if 2 * residue < PRIME {
            residue
        } else {
            PRIME - residue
        }
    }

    /// Generate an uniformly random integer from a given RNG.
    pub fn random(rng: &mut ChaCha8Rng) -> Self {
        BaseRingElem {
            element: MontgomeryInt::new(rng.random_range(0..PRIME), &PRIME),
        }
    }

    /// Convert a `BaseRingElem` into a `Vec<u8>` (least-endian)
    /// of size `PRIME_BYTES_LEN`.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.element.residue().to_le_bytes()[((DEGREE as usize >> 3) - PRIME_BYTES_LEN)..].to_vec()
    }

    pub fn from_le_bytes(bytes: &[u8; PRIME_BYTES_LEN]) -> Self {
        let mut sum: u64 = 0;
        for i in 0..PRIME_BYTES_LEN {
            // least endian: bytes[7] has smallest weight
            sum += (bytes[PRIME_BYTES_LEN - 1 - i] as u64) << (8 * i);
        }

        sum.into()
    }

    /// Divide by `2` modulo `p`.
    pub fn halve(&self) -> Self {
        let one_half: BaseRingElem = ONE_HALF_MOD_PRIME.into();
        self * one_half
    }
}

impl PolyRingElem {
    /// Generate the null polynomial.
    pub fn zero() -> Self {
        PolyRingElem {
            element: vec![BaseRingElem::zero(); DEGREE as usize],
        }
    }

    /// Generate the constant polynomial equal to 1.
    pub fn one() -> Self {
        let mut element = vec![BaseRingElem::zero(); DEGREE as usize];
        element[0] = BaseRingElem::one();

        PolyRingElem { element }
    }

    /// Compute the sum of each squared coefficients. In fact for each coefficient `coef`,
    /// take the minimum of `coef * coef` and `(PRIME - coef) * (PRIME - coef)`
    pub fn norm_square(&self) -> u128 {
        self.element
            .iter()
            .map(|b| {
                let abs = b.abs() as u128;
                abs * abs
            })
            .sum()
    }

    pub fn is_zero(&self) -> bool {
        self.element.iter().all(|coef| coef.element.is_zero())
    }

    // /// Divide by `2` modulo `p`.
    // pub fn halve(&self) -> Self {
    //     Self {
    //         element: self.element.iter().map(|c| c.halve()).collect(),
    //     }
    // }

    pub fn neg(&mut self) {
        self.element.iter_mut().for_each(|c| *c = -*c);
    }

    /// Multiply the polynomial by `X^exp`, in the ring where `X^DEGREE + 1 = 0`.
    pub fn mul_by_x_power(&self, mut exp: u64) -> Self {
        let mut result = Self::zero();

        let sign: BaseRingElem = if ((exp / DEGREE) & 1) == 1 {
            -BaseRingElem::one()
        } else {
            BaseRingElem::one()
        };

        exp %= DEGREE;

        let (pos_part, neg_part) = self.element.split_at((DEGREE - exp) as usize);

        let mut i = 0_usize;

        neg_part.iter().for_each(|coef| {
            result.element[i] = &-sign * coef;
            i += 1;
        });

        for coef in pos_part.iter() {
            result.element[i] = &sign * coef;
            i += 1;
        }

        result
    }

    /// Build a `PolyRingElem` from a list of coefficients.
    pub fn from_slice_u64(v: &[u64]) -> Self {
        Self {
            element: v.iter().map(|u| (*u).into()).collect(),
        }
    }

    /// Build a `PolyRingElem` from a list of coefficients.
    pub fn from_vec_u64(v: Vec<u64>) -> Self {
        Self {
            element: v.iter().map(|u| (*u).into()).collect(),
        }
    }

    /// Convert a `PolyRingElem` to a vector of bytes.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut result: Vec<u8> = Vec::new();

        for coef in &self.element {
            result.append(&mut coef.to_le_bytes());
        }

        result
    }

    pub fn hash(&self, output: &mut [u8]) {
        let mut hasher = Shake128::default();
        hasher.update(&self.to_le_bytes());

        let mut reader = hasher.finalize_xof();
        reader.read(output);
    }

    pub fn hash_many(input: &[PolyRingElem], output: &mut [u8]) {
        let mut hasher = Shake128::default();
        input
            .iter()
            .for_each(|poly| hasher.update(&poly.to_le_bytes()));

        let mut reader = hasher.finalize_xof();
        reader.read(output);
    }

    /// Generate an uniformly random polynomial from a given RNG.
    pub fn random(rng: &mut ChaCha8Rng) -> Self {
        Self {
            element: (0..DEGREE).map(|_| BaseRingElem::random(rng)).collect(),
        }
    }

    /// Generate a polynomial with 32 coefficients equal to `+-1`, 8
    /// coefficients equal to `+-2`, and 24 zero coefficients.
    pub fn challenge(rng: &mut ChaCha8Rng) -> Self {
        // generate signs
        let tau1_size = TAU1 as usize;
        let non_zero_size = tau1_size + TAU2 as usize;
        let bernoulli = Bernoulli::from_ratio(1, 2).unwrap();
        let signs: Vec<bool> = (0..non_zero_size).map(|_| bernoulli.sample(rng)).collect();

        // generate element
        let deg_size = DEGREE as usize;
        let mut element: Vec<BaseRingElem> = (0..tau1_size)
            .map(|i| {
                if signs[i] {
                    -BaseRingElem::from(1)
                } else {
                    1.into()
                }
            })
            .chain((tau1_size..non_zero_size).map(|i| {
                if signs[i] {
                    -BaseRingElem::from(2)
                } else {
                    2.into()
                }
            }))
            .chain((non_zero_size..deg_size).map(|_| 0.into()))
            .collect();

        element.shuffle(rng);

        Self { element }
    }

    /// Takes a slice of 32 `u8` to generate a RNG, then call [`PolyRingElem::Challenge`].
    pub fn challenge_from_seed(seed: &[u8]) -> Self {
        let mut owned_seed = [0u8; 32];
        owned_seed.copy_from_slice(seed);

        Self::challenge(&mut ChaCha8Rng::from_seed(owned_seed))
    }

    /// Decompose the polynomial in the base `2^base`, returns a list of `len` polynomials.
    pub fn decompose(&self, log_base: usize, len: usize) -> Vec<Self> {
        let base = 1 << log_base;

        let mut result: Vec<Vec<BaseRingElem>> = vec![Vec::new(); len];

        for i in 0..(DEGREE as usize) {
            let positive = 2 * self.element[i].element.residue() < PRIME;
            let mut coef = self.element[i].abs();

            if positive {
                for j in 0..len {
                    let coef_limb = coef & (base - 1);
                    result[j].push(coef_limb.into());
                    coef >>= log_base as u64;
                }
            } else {
                for j in 0..len {
                    let coef_limb = coef & (base - 1);
                    result[j].push(-BaseRingElem::from(coef_limb));
                    coef >>= log_base as u64;
                }
            }
        }

        result.into_iter().map(|v| Self { element: v }).collect()
    }

    /// Apply the ring automorphism defined by `sigma(X) = X^{-1} = -X^{d-1}` to the polynomial.
    pub fn invert_x(&mut self) {
        let deg_usize = DEGREE as usize;

        let mut new_elem: Vec<BaseRingElem> = Vec::with_capacity(deg_usize);
        new_elem.push(self.element[0]);

        for i in 1..((deg_usize - 1) / 2) {
            self.element.swap(i, deg_usize - i);
            if i & 1 == 1 {
                self.element[i] = -self.element[i];
            }
        }
        self.element = new_elem;
    }

    pub fn karatsuba_mul(max_deg: usize, left: &[BaseRingElem], right: &[BaseRingElem]) -> Self {
        if left.len() == 1 && right.len() == 1 {
            let mut result = PolyRingElem::zero();
            result.element[0] = &left[0] * &right[0];
            return result;
        }
        let mid_deg = max_deg.div_ceil(2);

        let (l1, l2) = left.split_at(mid_deg);
        let sum_l: Vec<BaseRingElem> = l1.iter().zip(l2.iter()).map(|(a, b)| a + b).collect();

        let (r1, r2) = right.split_at(mid_deg);
        let sum_r: Vec<BaseRingElem> = r1.iter().zip(r2.iter()).map(|(a, b)| a + b).collect();

        let small = Self::karatsuba_mul(mid_deg, l1, r1);
        let big = Self::karatsuba_mul(max_deg - mid_deg, l2, r2);
        let mut mid = Self::karatsuba_mul(mid_deg, &sum_l, &sum_r);
        mid -= &small;
        mid -= &big;

        let mut result: Self = small + mid.mul_by_x_power(mid_deg as u64);
        result += big.mul_by_x_power(2 * mid_deg as u64);

        result
    }
}

/*******************************************************************************
 * Add for BaseRingElem and PolyRingElem
 *******************************************************************************/

/// Add two `BaseRingElem`.
impl<T> Add<T> for &BaseRingElem
where
    T: AsRef<BaseRingElem>,
{
    type Output = BaseRingElem;
    fn add(self, other: T) -> BaseRingElem {
        let other: &BaseRingElem = other.as_ref();

        BaseRingElem {
            element: self.element + other.element,
        }
    }
}

/// Add two `BaseRingElem`.
impl<T> Add<T> for BaseRingElem
where
    T: AsRef<BaseRingElem>,
{
    type Output = BaseRingElem;
    fn add(self, other: T) -> BaseRingElem {
        &self + other
    }
}

/// Add two `PolyRingElem`. Add each pair of `&BaseRingElem`.
impl<T> Add<T> for &PolyRingElem
where
    T: AsRef<PolyRingElem>,
{
    type Output = PolyRingElem;
    fn add(self, other: T) -> PolyRingElem {
        let other_: &PolyRingElem = other.as_ref();

        PolyRingElem {
            element: self
                .element
                .iter()
                .zip(other_.element.iter())
                .map(|(l, r)| l + r)
                .collect(),
        }
    }
}

/// Add two `PolyRingElem`. Add each pair of `&BaseRingElem`.
impl<T> Add<T> for PolyRingElem
where
    T: AsRef<PolyRingElem>,
{
    type Output = PolyRingElem;
    fn add(self, other: T) -> PolyRingElem {
        &self + other
    }
}

/*******************************************************************************
 * AddAssign for BaseRingElem and PolyRingElem
 *******************************************************************************/

impl<T> AddAssign<T> for BaseRingElem
where
    T: AsRef<BaseRingElem>,
{
    fn add_assign(&mut self, other: T) {
        let other_: &BaseRingElem = other.as_ref();
        self.element = self.element + other_.element;
    }
}

impl<T> AddAssign<T> for PolyRingElem
where
    T: AsRef<PolyRingElem>,
{
    fn add_assign(&mut self, other: T) {
        let other_: &PolyRingElem = other.as_ref();
        self.element
            .iter_mut()
            .zip(other_.element.iter())
            .for_each(|(l, r)| {
                *l += r;
            });
    }
}

/*******************************************************************************
 * Neg, Sub for BaseRingElem and PolyRingElem
 *******************************************************************************/

impl Neg for BaseRingElem {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            element: -self.element,
        }
    }
}

impl Neg for PolyRingElem {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            element: self.element.iter().map(|coef| -*coef).collect(),
        }
    }
}

impl Neg for &PolyRingElem {
    type Output = PolyRingElem;
    fn neg(self) -> Self::Output {
        PolyRingElem {
            element: self.element.iter().map(|coef| -*coef).collect(),
        }
    }
}

/// Sub two `BaseRingElem`.
impl<T> Sub<T> for &BaseRingElem
where
    T: AsRef<BaseRingElem>,
{
    type Output = BaseRingElem;
    fn sub(self, other: T) -> BaseRingElem {
        let other: &BaseRingElem = other.as_ref();

        BaseRingElem {
            element: self.element - other.element,
        }
    }
}

/// Sub two `BaseRingElem`.
impl<T> Sub<T> for BaseRingElem
where
    T: AsRef<BaseRingElem>,
{
    type Output = BaseRingElem;
    fn sub(self, other: T) -> BaseRingElem {
        &self - other
    }
}

/// Sub two `PolyRingElem`. Sub each pair of `&BaseRingElem`.
impl<T> Sub<T> for &PolyRingElem
where
    T: AsRef<PolyRingElem>,
{
    type Output = PolyRingElem;
    fn sub(self, other: T) -> PolyRingElem {
        let other: &PolyRingElem = other.as_ref();

        PolyRingElem {
            element: self
                .element
                .iter()
                .zip(other.element.iter())
                .map(|(l, r)| l - r)
                .collect(),
        }
    }
}

/// Sub two `PolyRingElem`. Sub each pair of `&BaseRingElem`.
impl<T> Sub<T> for PolyRingElem
where
    T: AsRef<PolyRingElem>,
{
    type Output = PolyRingElem;
    fn sub(self, other: T) -> PolyRingElem {
        &self - other
    }
}

/*******************************************************************************
 * SubAssign for BaseRingElem and PolyRingElem
 *******************************************************************************/

impl<T> SubAssign<T> for BaseRingElem
where
    T: AsRef<BaseRingElem>,
{
    fn sub_assign(&mut self, other: T) {
        let other_: &BaseRingElem = other.as_ref();
        self.element = self.element - other_.element;
    }
}

impl<T> SubAssign<T> for PolyRingElem
where
    T: AsRef<PolyRingElem>,
{
    fn sub_assign(&mut self, other: T) {
        let other_: &PolyRingElem = other.as_ref();
        self.element
            .iter_mut()
            .zip(other_.element.iter())
            .for_each(|(l, r)| {
                *l -= r;
            });
    }
}

/*******************************************************************************
 * Mul for BaseRingElem and PolyRingElem
 *******************************************************************************/

impl<T> Mul<T> for &BaseRingElem
where
    T: AsRef<BaseRingElem>,
{
    type Output = BaseRingElem;
    fn mul(self, other: T) -> BaseRingElem {
        let other_ref: &BaseRingElem = other.as_ref();

        BaseRingElem {
            element: self.element * other_ref.element,
        }
    }
}

impl<T> Mul<T> for BaseRingElem
where
    BaseRingElem: Mul<T, Output = BaseRingElem>,
{
    type Output = BaseRingElem;
    fn mul(self, other: T) -> BaseRingElem {
        self * other
    }
}

/// Scalar multiplication `PolyRingElem x BaseRingElem`.
impl<T> Mul<T> for &PolyRingElem
where
    T: AsRef<BaseRingElem>,
{
    type Output = PolyRingElem;
    fn mul(self, other: T) -> PolyRingElem {
        PolyRingElem {
            element: self
                .element
                .iter()
                .map(|coef| other.as_ref() * coef)
                .collect(),
        }
    }
}

/// Scalar mul assign `PolyRingElem x BaseRingElem`.
impl MulAssign<&BaseRingElem> for PolyRingElem {
    fn mul_assign(&mut self, other: &BaseRingElem) {
        self.element.iter_mut().for_each(|b| *b = &*b * other);
    }
}

/// Scalar mul assign `PolyRingElem x BaseRingElem`.
impl MulAssign<BaseRingElem> for PolyRingElem {
    fn mul_assign(&mut self, other: BaseRingElem) {
        self.element.iter_mut().for_each(|b| *b = &*b * other);
    }
}

/// Scalar multiplication `BaseRingElem x PolyRingElem`.
impl Mul<&PolyRingElem> for &BaseRingElem {
    type Output = PolyRingElem;
    fn mul(self, rhs: &PolyRingElem) -> Self::Output {
        rhs * self
    }
}

/// Scalar multiplication `BaseRingElem x PolyRingElem`.
impl Mul<PolyRingElem> for &BaseRingElem {
    type Output = PolyRingElem;
    fn mul(self, rhs: PolyRingElem) -> Self::Output {
        self * &rhs
    }
}

/// Polynomial mul assign.
impl MulAssign<&PolyRingElem> for PolyRingElem {
    fn mul_assign(&mut self, other: &PolyRingElem) {
        // self.element =
        //     PolyRingElem::karatsuba_mul(DEGREE as usize, &self.element, &other.element).element;
        let clone = self.clone();
        self.element.fill(0.into());

        for (i, coef) in other.element.iter().enumerate() {
            let tmp = coef * clone.mul_by_x_power(i as u64);
            *self += tmp;
        }
    }
}

impl<T> Mul<T> for PolyRingElem
where
    PolyRingElem: MulAssign<T>,
{
    type Output = PolyRingElem;
    fn mul(self, other: T) -> Self::Output {
        let mut new = self.clone();
        new *= other;
        new
    }
}

/// Polynomial multiplication.
impl Mul<&PolyRingElem> for &PolyRingElem {
    type Output = PolyRingElem;
    fn mul(self, other: &PolyRingElem) -> Self::Output {
        // PolyRingElem::karatsuba_mul(DEGREE as usize, &self.element, &other.element)
        let mut result = PolyRingElem::zero();
        let clone = self.clone();

        for (i, coef) in other.element.iter().enumerate() {
            let tmp = coef * clone.mul_by_x_power(i as u64);
            result += tmp;
        }

        result
    }
}

/// Polynomial multiplication.
impl Mul<PolyRingElem> for &PolyRingElem {
    type Output = PolyRingElem;
    fn mul(self, other: PolyRingElem) -> Self::Output {
        self * &other
    }
}

/// Polynomial multiplication.
impl Mul<PolyRingElem> for PolyRingElem {
    type Output = PolyRingElem;
    fn mul(self, other: PolyRingElem) -> Self::Output {
        &self * &other
    }
}

#[cfg(test)]
mod tests {
    use crate::random_seed;

    use super::*;

    #[test]
    fn add_base() {
        let a: BaseRingElem = 2.into();
        let b: BaseRingElem = 131.into();

        assert_eq!(133, (a + b).element.residue());
    }

    #[test]
    fn wrapping_add_base() {
        let a: BaseRingElem = (PRIME - 2).into();
        let b: BaseRingElem = (1 << 10).into();

        assert_eq!(1022, (a + b).element.residue());
    }

    #[test]
    fn neg_base() {
        let a: BaseRingElem = 153.into();
        assert_eq!(PRIME - 153, (-a).element.residue());
    }

    #[test]
    fn mul_base() {
        let a: BaseRingElem = (PRIME / 10).into();
        let b: BaseRingElem = 257.into();
        let c: BaseRingElem = &a * &b;

        // computation done with ipython
        assert_eq!(769658139281, c.element.residue());
    }

    #[test]
    fn add_poly() {
        let c = PRIME - 3;
        let a = PolyRingElem::from_vec_u64(vec![
            c, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]);
        let b = PolyRingElem::from_vec_u64(vec![
            4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]);
        let result = PolyRingElem::from_vec_u64(vec![
            1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]);

        assert_eq!(a + b, result)
    }

    #[test]
    fn scalar_mul() {
        let scalar: BaseRingElem = 2.into();
        let a = PolyRingElem::from_vec_u64((1..65).collect());
        let result = PolyRingElem::from_vec_u64((1..65).map(|u| 2 * u).collect());

        assert_eq!(&scalar * &a, result);
    }

    #[test]
    fn pol_shift() {
        let a = PolyRingElem::from_vec_u64(vec![1; DEGREE as usize]);
        let b = a.mul_by_x_power(3);

        let c = PRIME - 1;

        assert_eq!(
            b,
            PolyRingElem::from_vec_u64(vec![
                c, c, c, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1
            ])
        );
    }

    #[test]
    fn pol_mul() {
        let a = PolyRingElem::from_vec_u64((0..64).collect());
        let b = PolyRingElem::from_vec_u64(vec![
            1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]);

        let c = a * b;

        let (c_1, c_2) = (PRIME - (2 * 63) - (3 * 62), 1 + PRIME - 3 * 63);
        let result = PolyRingElem::from_vec_u64(vec![
            c_1, c_2, 4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94, 100, 106, 112,
            118, 124, 130, 136, 142, 148, 154, 160, 166, 172, 178, 184, 190, 196, 202, 208, 214,
            220, 226, 232, 238, 244, 250, 256, 262, 268, 274, 280, 286, 292, 298, 304, 310, 316,
            322, 328, 334, 340, 346, 352, 358, 364, 370,
        ]);

        assert_eq!(result, c);
    }

    #[test]
    fn karatsuba_mul() {
        let a = PolyRingElem::from_vec_u64((0..64).collect());
        let b = PolyRingElem::from_vec_u64(vec![
            1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]);

        // let c = a * b;
        let c = PolyRingElem::karatsuba_mul(64, &a.element, &b.element);

        let (c_1, c_2) = (PRIME - (2 * 63) - (3 * 62), 1 + PRIME - 3 * 63);
        let result = PolyRingElem::from_vec_u64(vec![
            c_1, c_2, 4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94, 100, 106, 112,
            118, 124, 130, 136, 142, 148, 154, 160, 166, 172, 178, 184, 190, 196, 202, 208, 214,
            220, 226, 232, 238, 244, 250, 256, 262, 268, 274, 280, 286, 292, 298, 304, 310, 316,
            322, 328, 334, 340, 346, 352, 358, 364, 370,
        ]);

        assert_eq!(result, c);
    }

    #[test]
    fn two_mul_are_equal() {
        let seed = random_seed();
        let mut rng = ChaCha8Rng::from_seed(seed);
        let iterations: usize = 1 << 16;

        for i in 0..iterations {
            let left = PolyRingElem::random(&mut rng);
            let right = PolyRingElem::random(&mut rng);

            println!("{i}");

            assert_eq!(
                &left * &right,
                PolyRingElem::karatsuba_mul(DEGREE as usize, &left.element, &right.element)
            )
        }
    }

    #[test]
    fn pol_decomp() {
        let a = PolyRingElem::from_vec_u64((0..64).collect());

        let log_base = 2;
        let length = 5;
        let decomp = a.decompose(log_base, length);

        println!("decomp:\n{decomp:?}");

        let result_head = [
            PolyRingElem::from_vec_u64(vec![
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                0, 1, 2, 3, 0, 1, 2, 3,
            ]),
            PolyRingElem::from_vec_u64(vec![
                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1,
                2, 2, 2, 2, 3, 3, 3, 3,
            ]),
            PolyRingElem::from_vec_u64(vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                3, 3, 3, 3, 3, 3, 3, 3,
            ]),
        ];

        assert_eq!(decomp[0], result_head[0]);
        assert_eq!(decomp[1], result_head[1]);
        assert_eq!(decomp[2], result_head[2]);

        for pol in decomp[3..].into_iter() {
            assert_eq!(*pol, PolyRingElem::zero());
        }
    }
}

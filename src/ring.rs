use crate::constants::{DEGREE, PRIME};
use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// An element of `Z/pZ`, where `p = PRIME`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BaseRingElem {
    pub element: u64,
}

/// An element of `(Z/pZ)[X] / (X^d + 1)`, where `p = PRIME`, and `d = DEGREE`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyRingElem {
    pub element: Vec<BaseRingElem>,
}

/// Take residue mod `PRIME`.
impl From<u64> for BaseRingElem {
    fn from(value: u64) -> Self {
        Self {
            element: value % PRIME,
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
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.element)
    }
}

impl BaseRingElem {
    /// Generate 0 mod `PRIME`.
    pub fn zero() -> Self {
        BaseRingElem { element: 0 }
    }
    /// Generate 1 mod `PRIME`.
    pub fn one() -> Self {
        BaseRingElem { element: 1 }
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

    /// Multiply the polynomial by `X^exp`, in the ring where `X^DEGREE + 1 = 0`.
    pub fn mul_by_x_power(self, mut exp: u64) -> Self {
        let mut result = Self::zero();
        exp %= DEGREE;

        let (pos_part, neg_part) = self.element.split_at((DEGREE - exp) as usize);

        let mut i = 0_usize;

        for coef in neg_part.iter() {
            result.element[i] = -*coef;
            i += 1;
        }

        for coef in pos_part.iter() {
            result.element[i] = *coef;
            i += 1;
        }

        result
    }

    /// Build a `PolyRingElem` from a list of coefficients.
    pub fn from_vec_u64(v: Vec<u64>) -> Self {
        Self {
            element: v.iter().map(|u| (*u).into()).collect(),
        }
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
            element: (self.element + other.element) % PRIME,
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
        self.element += other_.element;
        self.element %= PRIME;
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
        if self.element == 0 {
            self
        } else {
            Self {
                element: PRIME - self.element,
            }
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

/// Sub two `BaseRingElem`.
impl<T> Sub<T> for &BaseRingElem
where
    T: AsRef<BaseRingElem>,
{
    type Output = BaseRingElem;
    fn sub(self, other: T) -> BaseRingElem {
        let other: &BaseRingElem = other.as_ref();

        if self.element >= other.element {
            BaseRingElem {
                element: self.element - other.element,
            }
        } else {
            self + -*other
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
        self.element += PRIME - other_.element;
        self.element %= PRIME;
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
        let other_: &BaseRingElem = other.as_ref();

        // the product of two u64 should be seen as a u128
        let result = (self.element as u128) * (other_.element as u128);
        BaseRingElem {
            element: (result % (PRIME as u128)) as u64,
        }
    }
}

impl<T> Mul<T> for BaseRingElem
where
    T: AsRef<BaseRingElem>,
{
    type Output = BaseRingElem;
    fn mul(self, other: T) -> BaseRingElem {
        &self * other
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

/// Scalar multiplication `PolyRingElem x BaseRingElem`.
impl<T> Mul<T> for PolyRingElem
where
    T: AsRef<BaseRingElem>,
{
    type Output = PolyRingElem;
    fn mul(self, other: T) -> PolyRingElem {
        &self * other
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
impl Mul<&PolyRingElem> for BaseRingElem {
    type Output = PolyRingElem;
    fn mul(self, rhs: &PolyRingElem) -> Self::Output {
        &self * rhs
    }
}

/// Scalar multiplication `BaseRingElem x PolyRingElem`.
impl Mul<PolyRingElem> for &BaseRingElem {
    type Output = PolyRingElem;
    fn mul(self, rhs: PolyRingElem) -> Self::Output {
        self * &rhs
    }
}

/// Scalar multiplication `BaseRingElem x PolyRingElem`.
impl Mul<PolyRingElem> for BaseRingElem {
    type Output = PolyRingElem;
    fn mul(self, rhs: PolyRingElem) -> Self::Output {
        &self * &rhs
    }
}

/// Polynomial multiplication.
impl Mul<&PolyRingElem> for &PolyRingElem {
    type Output = PolyRingElem;
    fn mul(self, other: &PolyRingElem) -> Self::Output {
        let mut result = PolyRingElem::zero();

        for (i, coef) in other.element.iter().enumerate() {
            let tmp = coef * self.clone().mul_by_x_power(i as u64);
            result += tmp;
        }

        result
    }
}

/// Polynomial multiplication.
impl Mul<&PolyRingElem> for PolyRingElem {
    type Output = PolyRingElem;
    fn mul(self, other: &PolyRingElem) -> Self::Output {
        &self * other
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
    use super::*;

    #[test]
    fn add_base() {
        let a: BaseRingElem = 2.into();
        let b: BaseRingElem = 131.into();

        assert_eq!(133, (a + b).element);
    }

    #[test]
    fn wrapping_add_base() {
        let a: BaseRingElem = (PRIME - 2).into();
        let b: BaseRingElem = (1 << 10).into();

        assert_eq!(1022, (a + b).element);
    }

    #[test]
    fn neg_base() {
        let a: BaseRingElem = 153.into();
        assert_eq!(PRIME - 153, (-a).element);
    }

    #[test]
    fn mul_base() {
        let a: BaseRingElem = (PRIME / 10).into();
        let b: BaseRingElem = 257.into();
        let c: BaseRingElem = a * b;

        // computation done with ipython
        assert_eq!(769658139281, c.element);
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

        assert_eq!(scalar * a, result);
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
}

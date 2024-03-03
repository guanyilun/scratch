use ndarray::Array1;
use std::ops::{Mul, MulAssign};

// trait Quaternion {
//     fn mul(&self, other: &Self) -> Self;
//     fn add(&self, other: &Self) -> Self;
//     fn conj(&self) -> Self;
// }

#[derive(Debug)]
struct Quat {
    w: f64,
    x: f64,
    y: f64,
    z: f64,
}

impl Quat {
    fn new(w: f64, x: f64, y: f64, z: f64) -> Quat {
        Quat { w, x, y, z }
    }
    fn conj(&self) -> Self {
        Quat {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// impl Quaternion for Quat {
//     fn mul(&self, q2: &Self) -> Self {
//         Self {
//             w: self.w * q2.w - self.x * q2.x - self.y * q2.y - self.z * q2.z,
//             x: self.w * q2.x + self.x * q2.w + self.y * q2.z - self.z * q2.y,
//             y: self.w * q2.y - self.x * q2.z + self.y * q2.w + self.z * q2.x,
//             z: self.w * q2.z + self.x * q2.y - self.y * q2.x + self.z * q2.w,
//         }
//     }

//     fn add(&self, other: &Self) -> Self {
//         Self {
//             w: self.w + other.w,
//             x: self.x + other.x,
//             y: self.y + other.y,
//             z: self.z + other.z,
//         }
//     }

//     }
// }

#[derive(Debug)]
struct QuatVec {
    w: Array1<f64>,
    x: Array1<f64>,
    y: Array1<f64>,
    z: Array1<f64>,
}

impl QuatVec {
    fn new(w: Array1<f64>, x: Array1<f64>, y: Array1<f64>, z: Array1<f64>) -> QuatVec {
        QuatVec { w, x, y, z }
    }
    fn conj(&self) -> Self {
        Self {
            w: self.w.clone(),
            x: -&self.x,
            y: -&self.y,
            z: -&self.z,
        }
    }
}

// impl Quaternion for QuatVec {
//     fn mul(&self, q2: &Self) -> Self {
//         Self {
//             w: &self.w * &q2.w - &self.x * &q2.x - &self.y * &q2.y - &self.z * &q2.z,
//             x: &self.w * &q2.x + &self.x * &q2.w + &self.y * &q2.z - &self.z * &q2.y,
//             y: &self.w * &q2.y - &self.x * &q2.z + &self.y * &q2.w + &self.z * &q2.x,
//             z: &self.w * &q2.z + &self.x * &q2.y - &self.y * &q2.x + &self.z * &q2.w,
//         }
//     }

//     fn add(&self, q2: &Self) -> Self {
//         Self {
//             w: &self.w + &q2.w,
//             x: &self.x + &q2.x,
//             y: &self.y + &q2.y,
//             z: &self.z + &q2.z,
//         }
//     }

// }

// macro_rules! impl_mul {
//     ($t:ty, $u:ty, $v:ident) => {
//         impl Mul<$u> for $t {
//             type Output = $v;

//             fn mul(self, q2: $u) -> $v {
//                 $v {
//                     w: self.w * q2.w - self.x * q2.x - self.y * q2.y - self.z * q2.z,
//                     x: self.w * q2.x + self.x * q2.w + self.y * q2.z - self.z * q2.y,
//                     y: self.w * q2.y - self.x * q2.z + self.y * q2.w + self.z * q2.x,
//                     z: self.w * q2.z + self.x * q2.y - self.y * q2.x + self.z * q2.w,
//                 }
//             }
//         }
//     };
// }

// impl_mul!(&Quat, &Quat, Quat);
// impl_mul!(&Quat, &QuatVec, QuatVec);
// impl_mul!(&QuatVec, &Quat, QuatVec);
// impl_mul!(&QuatVec, &QuatVec, QuatVec);

// implement &Qant * &Quat
impl Mul<&Quat> for &Quat {
    type Output = Quat;

    fn mul(self, q2: &Quat) -> Quat {
        Quat {
            w: self.w * q2.w - self.x * q2.x - self.y * q2.y - self.z * q2.z,
            x: self.w * q2.x + self.x * q2.w + self.y * q2.z - self.z * q2.y,
            y: self.w * q2.y - self.x * q2.z + self.y * q2.w + self.z * q2.x,
            z: self.w * q2.z + self.x * q2.y - self.y * q2.x + self.z * q2.w,
        }
    }
}

// implement &Quat * &QuatVec
impl Mul<&QuatVec> for &Quat {
    type Output = QuatVec;

    fn mul(self, q2: &QuatVec) -> QuatVec {
        QuatVec {  // &needed on q2 because q2.w are Array1<f64>
            w: self.w * &q2.w - self.x * &q2.x - self.y * &q2.y - self.z * &q2.z,
            x: self.w * &q2.x + self.x * &q2.w + self.y * &q2.z - self.z * &q2.y,
            y: self.w * &q2.y - self.x * &q2.z + self.y * &q2.w + self.z * &q2.x,
            z: self.w * &q2.z + self.x * &q2.y - self.y * &q2.x + self.z * &q2.w,
        }
    }
}

// implement &QuatVec * &QuatVec
impl Mul<&QuatVec> for &QuatVec {
    type Output = QuatVec;

    fn mul(self, q2: &QuatVec) -> QuatVec {
        QuatVec {
            w: &self.w * &q2.w - &self.x * &q2.x - &self.y * &q2.y - &self.z * &q2.z,
            x: &self.w * &q2.x + &self.x * &q2.w + &self.y * &q2.z - &self.z * &q2.y,
            y: &self.w * &q2.y - &self.x * &q2.z + &self.y * &q2.w + &self.z * &q2.x,
            z: &self.w * &q2.z + &self.x * &q2.y - &self.y * &q2.x + &self.z * &q2.w,
        }
    }
}

// implement &QuatVec * &Quat
impl Mul<&Quat> for &QuatVec {
    type Output = QuatVec;

    fn mul(self, q2: &Quat) -> QuatVec {
        QuatVec {
            w: &self.w * q2.w - &self.x * q2.x - &self.y * q2.y - &self.z * q2.z,
            x: &self.w * q2.x + &self.x * q2.w + &self.y * q2.z - &self.z * q2.y,
            y: &self.w * q2.y - &self.x * q2.z + &self.y * q2.w + &self.z * q2.x,
            z: &self.w * q2.z + &self.x * q2.y - &self.y * q2.x + &self.z * q2.w,
        }
    }
}

// implement Quat *= &Qant
impl MulAssign<&Quat> for Quat {
    fn mul_assign(&mut self, other: &Quat) {
        self.w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z;
        self.x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y;
        self.y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x;
        self.z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w; 
    }
}

// implement QuatVec *= &Qant
impl MulAssign<&Quat> for QuatVec {
    fn mul_assign(&mut self, other: &Quat) {
        self.w = &self.w * other.w - &self.x * other.x - &self.y * other.y - &self.z * other.z;
        self.x = &self.w * other.x + &self.x * other.w + &self.y * other.z - &self.z * other.y;
        self.y = &self.w * other.y - &self.x * other.z + &self.y * other.w + &self.z * other.x;
        self.z = &self.w * other.z + &self.x * other.y - &self.y * other.x + &self.z * other.w; 
    }
}

// implement QuatVec *= &QuatVec
impl MulAssign<&QuatVec> for QuatVec {
    fn mul_assign(&mut self, other: &QuatVec) {
        self.w = &self.w * &other.w - &self.x * &other.x - &self.y * &other.y - &self.z * &other.z;
        self.x = &self.w * &other.x + &self.x * &other.w + &self.y * &other.z - &self.z * &other.y;
        self.y = &self.w * &other.y - &self.x * &other.z + &self.y * &other.w + &self.z * &other.x;
        self.z = &self.w * &other.z + &self.x * &other.y - &self.y * &other.x + &self.z * &other.w; 
    }
}

fn main() {
    let mut q1 = Quat::new(1.0, 2.0, 3.0, 4.0);
    let q2 = Quat::new(5.0, 6.0, 7.0, 8.0);


    println!("{:?}", q1);
    println!("{:?}", q1.conj());
    println!("{:?}", q2.conj());
    println!("{:?}", &q1 * &q2);
    q1 *= &q2;
    println!("{:?}", q1);

    let mut qv1 = QuatVec::new(
        ndarray::array![1.0, 2.0, 3.0],
        ndarray::array![4.0, 5.0, 6.0],
        ndarray::array![7.0, 8.0, 9.0],
        ndarray::array![10.0, 11.0, 12.0],
    );
    println!("{:?}", qv1.conj());
    println!("{:?}", &qv1 * &q1);
    qv1 *= &q1;
    println!("{:?}", &qv1);
}

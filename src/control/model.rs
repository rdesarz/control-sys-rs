extern crate nalgebra as na;

pub struct ContinuousStateSpaceModel {
    mat_a: na::DMatrix<f64>,
    mat_b: na::DMatrix<f64>,
    mat_c: na::DMatrix<f64>,
    mat_d: na::DMatrix<f64>,
}

impl ContinuousStateSpaceModel {
    pub fn new(
        mat_a: &na::DMatrix<f64>,
        mat_b: &na::DMatrix<f64>,
        mat_c: &na::DMatrix<f64>,
        mat_d: &na::DMatrix<f64>,
    ) -> ContinuousStateSpaceModel {
        ContinuousStateSpaceModel {
            mat_a: mat_a.clone(),
            mat_b: mat_b.clone(),
            mat_c: mat_c.clone(),
            mat_d: mat_d.clone(),
        }
    }

    pub fn realize_from_tf(tf: &TransferFunction) -> ContinuousStateSpaceModel {
        let mat_a = na::dmatrix![0.0f64; 0.0f64];
        let mat_c = na::dmatrix![0.0f64; 0.0f64];
        

        let mut mat_b = na::DMatrix::<f64>::zeros(tf.denominator_coeffs.len(), 1);
        for (i, value) in tf.denominator_coeffs.iter().enumerate() {
            mat_b[(i, 0)] = value.clone();
        }

        let mat_d = na::dmatrix![tf.constant];

        ContinuousStateSpaceModel::new(&mat_a, &mat_b, &mat_c, &mat_d)
    }

    pub fn get_mat_a(&self) -> &na::DMatrix<f64> {
        return &self.mat_a;
    }

    pub fn get_mat_b(&self) -> &na::DMatrix<f64> {
        return &self.mat_b;
    }

    pub fn get_mat_c(&self) -> &na::DMatrix<f64> {
        return &self.mat_c;
    }

    pub fn get_mat_d(&self) -> &na::DMatrix<f64> {
        return &self.mat_d;
    }
}

pub trait DiscreteStateSpaceModel {
    fn get_mat_a(&self) -> &na::DMatrix<f64>;
    fn get_mat_b(&self) -> &na::DMatrix<f64>;
    fn get_mat_c(&self) -> &na::DMatrix<f64>;
    fn get_sampling_dt(&self) -> f64;
}

pub mod dc_motor {

    extern crate nalgebra as na;
    use crate::control::model::DiscreteStateSpaceModel;
    use std::default::Default;

    pub struct Parameters {
        b: f64,
        j: f64,
        k: f64,
        l: f64,
        r: f64,
    }

    impl Default for Parameters {
        fn default() -> Parameters {
            Parameters {
                b: 0.1,
                j: 0.01,
                k: 0.01,
                l: 0.5,
                r: 1.0,
            }
        }
    }

    #[derive(Clone)]
    pub struct Model {
        mat_a: na::DMatrix<f64>,
        mat_b: na::DMatrix<f64>,
        mat_c: na::DMatrix<f64>,
        sampling_dt: f64,
    }

    impl Model {
        pub fn new(params: Parameters, sampling_dt: f64) -> Model {
            // Define the continuous-time system matrices
            let mat_ac = na::dmatrix![
                -params.b / params.j, params.k / params.j;
                -params.k / params.l, -params.r / params.l;
            ];
            let mat_bc = na::dmatrix![0.0; 1.0 / params.l];
            let mat_cc = na::dmatrix![1.0, 0.0];

            // Model discretization
            let mat_i = na::DMatrix::<f64>::identity(mat_ac.nrows(), mat_ac.nrows());
            let mat_a = (mat_i - mat_ac.scale(sampling_dt)).try_inverse().unwrap();
            let mat_b = &mat_a * mat_bc.scale(sampling_dt);
            let mat_c = mat_cc;

            Model {
                mat_a: mat_a,
                mat_b: mat_b,
                mat_c: mat_c,
                sampling_dt: sampling_dt,
            }
        }
    }

    impl DiscreteStateSpaceModel for Model {
        fn get_mat_a(&self) -> &na::DMatrix<f64> {
            &self.mat_a
        }

        fn get_mat_b(&self) -> &na::DMatrix<f64> {
            &self.mat_b
        }

        fn get_mat_c(&self) -> &na::DMatrix<f64> {
            &self.mat_c
        }

        fn get_sampling_dt(&self) -> f64 {
            self.sampling_dt
        }
    }
}

pub mod mpc {
    extern crate nalgebra as na;

    use std::default::Default;

    use crate::control::model::DiscreteStateSpaceModel;

    pub struct Parameters {
        m1: f64,
        m2: f64,
        k1: f64,
        k2: f64,
        d1: f64,
        d2: f64,
    }

    impl Default for Parameters {
        fn default() -> Parameters {
            Parameters {
                m1: 2.0,
                m2: 2.0,
                k1: 100.0,
                k2: 200.0,
                d1: 1.0,
                d2: 5.0,
            }
        }
    }

    #[derive(Clone)]
    pub struct Model {
        mat_a: na::DMatrix<f64>,
        mat_b: na::DMatrix<f64>,
        mat_c: na::DMatrix<f64>,
        sampling_dt: f64,
    }

    impl Model {
        pub fn new(params: Parameters, sampling_dt: f64) -> Model {
            // Define the continuous-time system matrices
            let mat_ac = na::dmatrix![
                0.0, 1.0, 0.0, 0.0;
                -(params.k1 + params.k2) / params.m1, -(params.d1 + params.d2) / params.m1, params.k2 / params.m1, params.d2 / params.m1;
                0.0, 0.0, 0.0, 1.0;
                params.k2 / params.m2, params.d2 / params.m2, -params.k2 / params.m2, -params.d2 / params.m2
            ];
            let mat_bc = na::dmatrix![0.0; 0.0; 0.0; 1.0 / params.m2];
            let mat_cc = na::dmatrix![1.0, 0.0, 0.0, 0.0];

            // Model discretization
            let mat_i = na::DMatrix::<f64>::identity(mat_ac.nrows(), mat_ac.nrows());
            let mat_a = (mat_i - mat_ac.scale(sampling_dt)).try_inverse().unwrap();
            let mat_b = &mat_a * mat_bc.scale(sampling_dt);
            let mat_c = mat_cc;

            Model {
                mat_a: mat_a,
                mat_b: mat_b,
                mat_c: mat_c,
                sampling_dt: sampling_dt,
            }
        }
    }

    impl DiscreteStateSpaceModel for Model {
        fn get_mat_a(&self) -> &na::DMatrix<f64> {
            &self.mat_a
        }

        fn get_mat_b(&self) -> &na::DMatrix<f64> {
            &self.mat_b
        }

        fn get_mat_c(&self) -> &na::DMatrix<f64> {
            &self.mat_c
        }

        fn get_sampling_dt(&self) -> f64 {
            self.sampling_dt
        }
    }
}

pub struct TransferFunction {
    pub numerator_coeffs: Vec<f64>,
    pub denominator_coeffs: Vec<f64>,
    constant: f64,
}

impl TransferFunction {
    pub fn new(
        numerator_coeffs: &[f64],
        denominator_coeffs: &[f64],
        constant: f64,
    ) -> TransferFunction {
        TransferFunction {
            numerator_coeffs: numerator_coeffs.to_vec(),
            denominator_coeffs: denominator_coeffs.to_vec(),
            constant: constant,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_state_space_model() {
        let tf = TransferFunction::new(&[1.0, 2.0, 3.0], &[1.0, 4.0, 6.0], 0.0);

        let ss_model = ContinuousStateSpaceModel::from_transfer_function(&tf);
    }
}

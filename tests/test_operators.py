import numpy as np

from tsaime import (
    covariance_weighted_inverse,
    fit_inverse_operator,
    marginal_cross_correlation,
    scalar_aime_operator,
)


def test_scalar_operator_keeps_v02_signature_and_equals_correlation() -> None:
    generator = np.random.default_rng(4)
    inputs = generator.normal(size=(500, 4))
    output = 0.8 * inputs[:, 0] - 0.3 * inputs[:, 2] + generator.normal(scale=0.2, size=500)
    operator, local = scalar_aime_operator(inputs, output)
    expected = np.asarray([
        np.corrcoef(inputs[:, index], output)[0, 1]
        for index in range(inputs.shape[1])
    ])
    np.testing.assert_allclose(operator, expected, atol=1e-12)
    assert local.shape == operator.shape


def test_scalar_operator_reports_missing_and_constant_columns_as_undefined() -> None:
    inputs = np.column_stack(
        [np.arange(10, dtype=float), np.ones(10), np.arange(10, dtype=float)]
    )
    inputs[3, 2] = np.nan
    operator, local = scalar_aime_operator(inputs, np.arange(10, dtype=float))
    assert np.isfinite(operator[0])
    assert np.isnan(operator[1:]).all()
    assert np.isnan(local[1:]).all()


def test_vector_operator_recovers_covariance_weighted_truth() -> None:
    generator = np.random.default_rng(9)
    dimension, outputs, n = 6, 3, 10000
    raw = generator.normal(size=(dimension, dimension))
    sigma = raw @ raw.T + 0.8 * np.eye(dimension)
    inputs_matrix = generator.multivariate_normal(np.zeros(dimension), sigma, size=n)
    forward = generator.normal(size=(outputs, dimension))
    output_matrix = inputs_matrix @ forward.T + generator.normal(scale=0.01, size=(n, outputs))
    fitted = fit_inverse_operator(inputs_matrix, output_matrix, ridge=0.01)
    x_z = fitted.x_scaler.transform(inputs_matrix)
    forward_z = (
        np.diag(1.0 / fitted.y_scaler.scale)
        @ forward
        @ np.diag(fitted.x_scaler.scale)
    )
    sigma_z = np.cov(x_z, rowvar=False, ddof=0)
    truth = covariance_weighted_inverse(forward_z, sigma_z, ridge=0.01)
    relative_error = np.linalg.norm(fitted.operator - truth) / np.linalg.norm(truth)
    assert relative_error < 0.02


def test_covariance_weighted_inverse_includes_output_noise() -> None:
    forward = np.eye(2)
    covariance = np.eye(2)
    noiseless = covariance_weighted_inverse(forward, covariance)
    noisy = covariance_weighted_inverse(
        forward,
        covariance,
        output_noise_covariance=3.0 * np.eye(2),
    )
    np.testing.assert_allclose(noiseless, np.eye(2))
    np.testing.assert_allclose(noisy, 0.25 * np.eye(2))


def test_vector_operator_is_not_marginal_correlation_for_dependent_outputs() -> None:
    generator = np.random.default_rng(12)
    inputs = generator.normal(size=(2000, 5))
    output_1 = inputs[:, 0] + 0.7 * inputs[:, 1]
    output_2 = output_1 + 0.15 * inputs[:, 2]
    outputs = np.column_stack([output_1, output_2])
    inverse = fit_inverse_operator(inputs, outputs, ridge=0.01).operator
    marginal = marginal_cross_correlation(inputs, outputs)
    assert np.linalg.norm(inverse - marginal) > 0.5


def test_reconstruct_returns_original_input_units() -> None:
    generator = np.random.default_rng(20)
    outputs = generator.normal(size=(1000, 2))
    inputs = outputs @ np.asarray([[2.0, -1.0, 0.5], [0.2, 1.4, -0.7]]) + np.asarray([10.0, 30.0, -5.0])
    model = fit_inverse_operator(inputs, outputs, ridge=0.0)
    reconstructed = model.reconstruct(outputs)
    np.testing.assert_allclose(reconstructed, inputs, atol=1e-10)


def test_vector_operator_satisfies_regularized_normal_equation() -> None:
    generator = np.random.default_rng(21)
    inputs = generator.normal(size=(1500, 6))
    outputs = np.column_stack(
        [
            inputs[:, 0] + 0.6 * inputs[:, 1],
            inputs[:, 0] - 0.4 * inputs[:, 2],
            0.5 * inputs[:, 3] + 0.2 * inputs[:, 4],
        ]
    ) + generator.normal(scale=0.1, size=(1500, 3))
    ridge = 0.03
    model = fit_inverse_operator(inputs, outputs, ridge=ridge)
    residual = (
        model.operator
        @ (model.output_covariance + ridge * np.eye(outputs.shape[1]))
        - model.cross_covariance
    )
    assert np.linalg.norm(residual) / np.linalg.norm(model.cross_covariance) < 1e-10

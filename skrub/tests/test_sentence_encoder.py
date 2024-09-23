import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import clone

from skrub import SentenceEncoder
from skrub._on_each_column import RejectColumn
from skrub._sentence_encoder import ModelNotFound

pytest.importorskip("sentence_transformers")


@pytest.fixture
def encoder():
    """Create a fixture such that we set the common parameters of the encoder.

    Here, we have two constraints:

    - We want a really small model to not blow up the memory of the GitHub Actions
      workers.
    - We want to force the use of the CPU. GitHub Actions workers on macOS ARM64 will
      detect the MPS backend, but due to limitations, no memory can be allocated.
      See https://github.com/actions/runner-images/issues/9918 for more details.
    """
    return SentenceEncoder(
        model_name="sentence-transformers/paraphrase-albert-small-v2",
        device="cpu",
    )


def test_missing_import_error(encoder):
    try:
        import sentence_transformers  # noqa
    except ImportError:
        pass
    else:
        return

    st = clone(encoder)
    x = pd.Series(["oh no"])
    with pytest.raises(ImportError, match="Missing optional dependency"):
        st.fit(x)


def test_sentence_encoder(df_module, encoder):
    X = df_module.make_column("", ["hello sir", "hola que tal"])
    encoder = clone(encoder).set_params(n_components=2)
    X_out = encoder.fit_transform(X)
    assert X_out.shape == (2, 2)

    X_out_2 = encoder.fit_transform(X)
    df_module.assert_frame_equal(X_out, X_out_2)


@pytest.mark.parametrize("X", [["hello"], "hello"])
def test_not_a_series(X, encoder):
    with pytest.raises(ValueError):
        clone(encoder).fit(X)


def test_not_a_series_with_string(df_module, encoder):
    X = df_module.make_column("", [1, 2, 3])
    with pytest.raises(RejectColumn):
        clone(encoder).fit(X)


def test_missing_value(df_module, encoder):
    X = df_module.make_column("", [None, None, "hey"])
    encoder = clone(encoder).set_params(n_components=None)
    X_out = encoder.fit_transform(X)

    assert X_out.shape == (3, 768)
    X_out = X_out.to_numpy()
    assert_array_equal(X_out[0, :], X_out[1, :])


def test_n_components(df_module, encoder):
    X = df_module.make_column("", ["hello sir", "hola que tal"])
    encoder_all = clone(encoder).set_params(n_components=None).fit(X)
    for meth in ("fit_transform", "transform"):
        X_out = getattr(encoder_all, meth)(X)
        assert X_out.shape[1] == 768
        assert encoder_all.n_components_ == 768

    encoder_2 = clone(encoder).set_params(n_components=2).fit(X)
    for meth in ("fit_transform", "transform"):
        X_out = getattr(encoder_2, meth)(X)
        assert X_out.shape[1] == 2
        assert encoder_2.n_components_ == 2

    encoder_30 = clone(encoder).set_params(n_components=30)
    with pytest.warns(UserWarning, match="The embeddings will be truncated"):
        X_out = encoder_30.fit_transform(X)
    assert not hasattr(encoder_30, "pca_")
    assert X_out.shape[1] == 30
    assert encoder_30.n_components_ == 30


def test_wrong_parameters(encoder):
    with pytest.raises(ValueError, match="Got n_components='yes'"):
        clone(encoder).set_params(n_components="yes")._check_params()

    with pytest.raises(ValueError, match="Got batch_size=-10"):
        clone(encoder).set_params(batch_size=-10)._check_params()

    with pytest.raises(ValueError, match="Got model_name=1"):
        clone(encoder).set_params(model_name=1)._check_params()

    with pytest.raises(ValueError, match="Got cache_folder=1"):
        clone(encoder).set_params(cache_folder=1)._check_params()

    with pytest.raises(ValueError, match="Got model_name=1"):
        clone(encoder).set_params(model_name=1)._check_params()


def test_wrong_model_name(encoder):
    x = pd.Series(["Good evening Dave"])
    with pytest.raises(ModelNotFound):
        clone(encoder).set_params(model_name="HAL-9000").fit(x)


def test_transform_equal_fit_transform(df_module, encoder):
    x = df_module.make_column("", ["hello again"])
    encoder = clone(encoder).set_params(n_components=None)
    X_out = encoder.fit_transform(x)
    X_out_2 = encoder.transform(x)
    df_module.assert_frame_equal(X_out, X_out_2)


def test_transform_error_on_float_data(df_module, encoder):
    """Check that we raise an error when data without any string is passed at
    transform."""
    x = df_module.make_column("", [1.0, 2.5, 3.7])

    encoder = clone(encoder).set_params(n_components=None)
    encoder.fit(df_module.make_column("", ["hello", "world"]))

    with pytest.raises(RejectColumn, match="does not contain strings"):
        encoder.transform(x)

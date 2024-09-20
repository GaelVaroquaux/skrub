import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from skrub import SentenceEncoder
from skrub._on_each_column import RejectColumn
from skrub._sentence_encoder import ModelNotFound

pytest.importorskip("sentence_transformers")

MODEL_NAME = "sentence-transformers/paraphrase-albert-small-v2"


def test_missing_import_error():
    try:
        import sentence_transformers  # noqa
    except ImportError:
        pass
    else:
        return

    st = SentenceEncoder(model_name_or_path=MODEL_NAME, device="cpu")
    x = pd.Series(["oh no"])
    with pytest.raises(ImportError, match="Missing optional dependency"):
        st.fit(x)


def test_sentence_encoder(df_module):
    X = df_module.make_column("", ["hello sir", "hola que tal"])
    encoder = SentenceEncoder(
        model_name_or_path=MODEL_NAME, n_components=2, device="cpu"
    )
    X_out = encoder.fit_transform(X)
    assert X_out.shape == (2, 2)

    X_out_2 = encoder.fit_transform(X)
    df_module.assert_frame_equal(X_out, X_out_2)


@pytest.mark.parametrize("X", [["hello"], "hello"])
def test_not_a_series(X):
    with pytest.raises(ValueError):
        SentenceEncoder(model_name_or_path=MODEL_NAME, device="cpu").fit(X)


def test_not_a_series_with_string(df_module):
    X = df_module.make_column("", [1, 2, 3])
    with pytest.raises(RejectColumn):
        SentenceEncoder(model_name_or_path=MODEL_NAME, device="cpu").fit(X)


def test_missing_value(df_module):
    X = df_module.make_column("", [None, None, "hey"])
    encoder = SentenceEncoder(
        model_name_or_path=MODEL_NAME, n_components="all", device="cpu"
    )
    X_out = encoder.fit_transform(X)

    assert X_out.shape == (3, 384)
    X_out = X_out.to_numpy()
    assert_array_equal(X_out[0, :], X_out[1, :])


def test_n_components(df_module):
    X = df_module.make_column("", ["hello sir", "hola que tal"])
    encoder = SentenceEncoder(
        model_name_or_path=MODEL_NAME, n_components="all", device="cpu"
    )
    X_out = encoder.fit_transform(X)
    assert X_out.shape[1] == 384
    assert encoder.n_components_ == 384

    encoder = SentenceEncoder(
        model_name_or_path=MODEL_NAME, n_components=2, device="cpu"
    )
    X_out = encoder.fit_transform(X)
    assert X_out.shape[1] == 2
    assert encoder.n_components_ == 2

    encoder = SentenceEncoder(
        model_name_or_path=MODEL_NAME, n_components=30, device="cpu"
    )
    with pytest.warns(UserWarning):
        X_out = encoder.fit_transform(X)
    assert not hasattr(encoder, "pca_")
    assert X_out.shape[1] == 30
    assert encoder.n_components_ == 30


def test_wrong_parameters():
    with pytest.raises(ValueError, match="Got n_components='yes'"):
        SentenceEncoder(
            model_name_or_path=MODEL_NAME, n_components="yes", device="cpu"
        )._check_params()

    with pytest.raises(ValueError, match="Got batch_size=-10"):
        SentenceEncoder(
            model_name_or_path=MODEL_NAME, batch_size=-10, device="cpu"
        )._check_params()

    with pytest.raises(ValueError, match="Got model_name_or_path=1"):
        SentenceEncoder(model_name_or_path=1, device="cpu")._check_params()

    with pytest.raises(ValueError, match="Got norm=l3"):
        SentenceEncoder(
            model_name_or_path=MODEL_NAME, norm="l3", device="cpu"
        )._check_params()

    with pytest.raises(ValueError, match="Got cache_folder=1"):
        SentenceEncoder(
            model_name_or_path=MODEL_NAME, cache_folder=1, device="cpu"
        )._check_params()

    with pytest.raises(ValueError, match="Got model_name_or_path=1"):
        SentenceEncoder(model_name_or_path=1, device="cpu")._check_params()


def test_wrong_model_name():
    x = pd.Series(["Good evening Dave"])
    with pytest.raises(ModelNotFound):
        SentenceEncoder(model_name_or_path="HAL-9000", device="cpu").fit(x)


def test_transform_equal_fit_transform(df_module):
    x = df_module.make_column("", ["hello again"])
    encoder = SentenceEncoder(model_name_or_path=MODEL_NAME, device="cpu")
    X_out = encoder.fit_transform(x)
    X_out_2 = encoder.transform(x)
    df_module.assert_frame_equal(X_out, X_out_2)

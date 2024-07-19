import pytest
from lm_eval.api.model import CipherLM
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate
from lm_eval.api.instance import Instance

class SimpleCipher:
    def encrypt(self, text):
        return ''.join(chr((ord(c) + 1) % 256) for c in text)

    def decrypt(self, text):
        return ''.join(chr((ord(c) - 1) % 256) for c in text)

@pytest.fixture
def cipher_lm():
    base_lm = HFLM(pretrained="gpt2", device="cpu")
    cipher = SimpleCipher()
    return CipherLM(base_lm, cipher.encrypt, cipher.decrypt)

def test_cipher_loglikelihood(cipher_lm):
    requests = [
        Instance(request_type="loglikelihood", doc={}, arguments=("Hello", " world"), idx=0),
        Instance(request_type="loglikelihood", doc={}, arguments=("How are", " you"), idx=1),
    ]
    results = cipher_lm.loglikelihood(requests)
    assert len(results) == 2
    for result in results:
        assert isinstance(result[0], float)
        assert isinstance(result[1], bool)

def test_cipher_generate_until(cipher_lm):
    requests = [
        Instance(request_type="generate_until", doc={}, arguments=("Once upon a time", "\n"), idx=0),
        Instance(request_type="generate_until", doc={}, arguments=("The quick brown", " fox"), idx=1),
    ]
    results = cipher_lm.generate_until(requests)
    assert len(results) == 2
    for result in results:
        assert isinstance(result, str)

def test_cipher_loglikelihood_rolling(cipher_lm):
    requests = [
        Instance(request_type="loglikelihood_rolling", doc={}, arguments=("This is a test",), idx=0),
        Instance(request_type="loglikelihood_rolling", doc={}, arguments=("Another test sentence",), idx=1),
    ]
    results = cipher_lm.loglikelihood_rolling(requests)
    assert len(results) == 2
    for result in results:
        assert isinstance(result, float)

def test_simple_evaluate_with_cipher():
    results = simple_evaluate(
        model="hf",
        model_args="pretrained=gpt2",
        tasks=["hellaswag"],
        num_fewshot=0,
        batch_size=1,
        cipher=SimpleCipher(),
    )
    assert "results" in results
    assert "hellaswag" in results["results"]
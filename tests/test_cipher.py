import pytest
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate
from lm_eval.api.instance import Instance

class SimpleCipher:
    def encrypt(self, text):
        return ''.join(chr((ord(c) + 1) % 256) for c in text)

    def decrypt(self, text):
        return ''.join(chr((ord(c) - 1) % 256) for c in text)

class CipherLM(LM):
    def __init__(self, base_lm: LM, encrypt: callable, decrypt: callable):
        super().__init__()
        self.base_lm = base_lm
        self.encrypt = encrypt
        self.decrypt = decrypt

    def loglikelihood(self, requests):
        encrypted_requests = []
        for req in requests:
            context, continuation = req.args
            encrypted_context = self.encrypt(context)
            encrypted_continuation = self.encrypt(continuation)
            encrypted_req = req._replace(args=(encrypted_context, encrypted_continuation))
            encrypted_requests.append(encrypted_req)

        results = self.base_lm.loglikelihood(encrypted_requests)
        return results

    def loglikelihood_rolling(self, requests):
        encrypted_requests = []
        for req in requests:
            context, = req.args
            encrypted_context = self.encrypt(context)
            encrypted_req = req._replace(args=(encrypted_context,))
            encrypted_requests.append(encrypted_req)

        results = self.base_lm.loglikelihood_rolling(encrypted_requests)
        return results

    def generate_until(self, requests):
        encrypted_requests = []
        for req in requests:
            context, until = req.args
            encrypted_context = self.encrypt(context)
            encrypted_req = req._replace(args=(encrypted_context, until))
            encrypted_requests.append(encrypted_req)

        encrypted_results = self.base_lm.generate_until(encrypted_requests)
        decrypted_results = [self.decrypt(result) for result in encrypted_results]
        return decrypted_results

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
        device="cpu",
        cipher=SimpleCipher(),
    )
    assert "results" in results
    assert "hellaswag" in results["results"]
"""T-201: llama.cpp inference engine with GBNF grammar-constrained decoding."""
import json
import time
import logging
from llama_cpp import Llama, LlamaGrammar

logger = logging.getLogger("InferenceEngine")

SYSTEM_PROMPT = (
    "You are a 3GPP Protocol Expert specializing in Root Cause Analysis.\n"
    "Analyze the logs provided.\n"
    "1. Ignore routine Heartbeats and KeepAlives.\n"
    "2. Filter out sympathetic noise (consequential failures).\n"
    "3. Identify the primary causal chain (NAS, NGAP, or RRC).\n"
    "4. Output ONLY valid JSON according to the schema."
)


class TelcoInferenceEngine:
    """Singleton wrapper around llama-cpp-python with GBNF grammar."""

    _instance = None

    @classmethod
    def get_instance(cls, model_path: str, grammar_path: str) -> "TelcoInferenceEngine":
        if cls._instance is None:
            cls._instance = cls(model_path, grammar_path)
        return cls._instance

    def __init__(self, model_path: str, grammar_path: str):
        logger.info(f"Loading model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=16384,
            n_gpu_layers=-1,
            verbose=False,
        )
        self.grammar = LlamaGrammar.from_file(grammar_path)
        logger.info("Model loaded, grammar initialized ✓")

    def analyze(self, raw_logs: str) -> dict:
        """Run semantic filtering on raw 3GPP logs. Returns events + metadata."""
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{raw_logs}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        raw_bytes = len(raw_logs.encode("utf-8"))
        start = time.time()

        try:
            output = self.llm(
                prompt,
                grammar=self.grammar,
                max_tokens=512,
                temperature=0.1,
                repeat_penalty=1.2,
                top_p=0.9,
                stop=["<|im_end|>"],
            )
            text = output["choices"][0]["text"].strip()
            latency = round(time.time() - start, 2)
            events = json.loads(text).get("events", [])
            return {"events": events, "raw_log_bytes": raw_bytes, "inference_latency_sec": latency}
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return {
                "events": [],
                "status": "error",
                "message": str(e),
                "raw_log_bytes": raw_bytes,
                "inference_latency_sec": round(time.time() - start, 2),
            }

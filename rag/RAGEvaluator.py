import pandas as pd
import ollama
import numpy as np
import pickle
from pydantic import BaseModel, Field
from typing import List
from tqdm import tqdm
import json

from labeled_dataset.utils_labeled_dataset import topic_names as topic_taxonomy
topic_taxonomy.append("None")  # add None to avoid forcibly assigning sentences to topics


# Define structured response format
class LLMResponse_RAG(BaseModel):
    """
    Needed to generate structured output from LLM responses
    """
    topic: str = Field(..., enum=topic_taxonomy)
    explanation: str
    probability: int = Field(..., ge=0, le=100) # Expecting 0-100%r


EMPTY_RESPONSE = LLMResponse_RAG(topic="Parsing Error", explanation="Parsing Error", probability=0)

# ---- New batch wrapper model ----
from pydantic import RootModel

class LLMBatchResponse(RootModel):
    root: list[LLMResponse_RAG]


class RAGEvaluator:
    def __init__(self, prompts, indices_to_check, sentences, output_path:str):
        """
        class to perform the RAG evaluation on the critical sentences 
        """
        self.prompts = prompts
        self.indices_to_check = indices_to_check
        self.output_path = output_path
        self.sentences = sentences
        

    def llm_answers(self, model, message, show_prompts=False):
        """
        Generate answers with LLM using Ollama.
        model: name of ollama model
        message: the user message / prompt
        show_prompts: if True, prints the LLM response
        """
    
        # Only user message is needed; system prompt is already baked into the model
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": message}],
            options={"temperature": 0.0},  # factual responses
            format=LLMResponse_RAG.model_json_schema()  # enforce JSON
        )
    
        raw = response["message"]["content"]
        try:
            answer = LLMResponse_RAG.model_validate_json(raw)
        except Exception as e:
            print(f"Error parsing response: {e}")
            answer = EMPTY_RESPONSE
    
        if show_prompts:
            print(answer)
    
        return answer


    def transform_answers(self, answers):
        """
        Transform answers into a pickle-safe structure with index for reverse traceability.
        `answers` must: [(sentence_index, LLMResponse_RAG | None), ...]
        """
        records = []
        for sentence_index, sentence, prompt, response in answers:
            records.append({
                "sentence_index": sentence_index,
                "sentence": sentence,
                "prompt": prompt,
                "topic": response.topic if response else None,
                "explanation": response.explanation if response else None,
                "probability": response.probability if response else None
            })
        return records


    # Save helpers
    def _append_record(self, record: dict):
        """
        Append one result line as JSONL (newline-delimited JSON)
        """
        with open(self.output_path + ".jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def run_parallel(
        self,
        model,
        show_prompt=False,
        test_prompts: int | None = None,
        max_workers: int = 32
    ):
        """
        ✅ Parallel RAG evaluation with:
          - sentence index
          - original sentence
          - generated prompt
          - structured LLM answer (taxonomy-safe)
          - no misalignment of (index ↔ prompt ↔ response)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
    
        # Optional test subset slicing (alignment preserved!)
        if test_prompts is not None:
            prompts = self.prompts[:test_prompts]
            indices_to_check = self.indices_to_check[:test_prompts]
        else:
            prompts = self.prompts
            indices_to_check = self.indices_to_check
    
        # 4️⃣ Run LLM evaluation in parallel, remembering (index ↔ prompt)
        answers = []
    
        def process_prompt(prompt_text):
            """Helper for thread execution"""
            return self.llm_answers(
                model=model,
                message=prompt_text,
                show_prompts=show_prompt
            )
    
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 🔹🔧 CHANGE APPLIED HERE:
            # Futures now remember BOTH the real sentence index and its own prompt string
            # (instead of only remembering index, which caused errors)
            futures = {
                executor.submit(process_prompt, prompt_text): (idx, prompt_text)
                for prompt_text, idx in zip(prompts, indices_to_check)
            }
    
            for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Progress"):
                sentence_idx, prompt_text = futures[future]  # unpack remembered pair
                # Recover the real original sentence ✅ (kept aligned!)
                sentence = self.sentences[sentence_idx]
    
                try:
                    result = future.result()
    
                    # 🔹🔧 CHANGE APPLIED HERE:
                    # Guarantee result is always a valid model (never a string / null object)
                    if result is None or not hasattr(result, "topic"):
                        result = EMPTY_RESPONSE  # safe fallback model
    
                    answers.append((sentence_idx, sentence, prompt_text, result))
    
                except Exception as e:
                    print(f"Error processing sentence {sentence_idx}: {e}")
                    answers.append((sentence_idx, sentence, prompt_text, EMPTY_RESPONSE))

                
                # Build record dict
                record = {
                    "sentence_index": sentence_idx,
                    "sentence": sentence,
                    "prompt": prompt_text,
                    "topic": result.topic,
                    "explanation": result.explanation,
                    "probability": result.probability
                }
            
                # Append to JSONL immediately
                self._append_record(record)
            
                # Keep in memory if needed for returning at the end
                answers.append((sentence_idx, sentence, prompt_text, result))
    
        # 5️⃣ Convert to pickle-safe dict records including the prompt text for reverse traceability
        # Transform now receives 4 values per tuple
        records = self.transform_answers(answers)
    

        # Stream-save periodically for large runs
        #if i % (batch_size * 10) == 0:
         #   self._save_partial(results)

        #self._save_final(results)
    
        return records

    
    def run(self, model: str, show_prompt: bool = False, test_prompts: int | None = None):
        """
        Serial run over prompts (useful for debugging or when parallelism causes issues).
        """
        if test_prompts is not None:
            prompts = self.prompts[:test_prompts]
            indices = self.indices_to_check[:test_prompts]
        else:
            prompts = self.prompts
            indices = self.indices_to_check

        answers = []
        
        # convert zip to list for a proper tqdm total
        pairs = list(zip(indices, prompts))
        for sentence_idx, prompt_text in tqdm(pairs, desc="LLM Progress"):
            sentence_text = self.sentences[sentence_idx]
            try:
                result = self.llm_answers(model=model, message=prompt_text, show_prompts=show_prompt)
                if result is None or not hasattr(result, "topic"):
                    result = EMPTY_RESPONSE
            except Exception as e:
                print(f"Error processing sentence {sentence_idx}: {e}")
                result = EMPTY_RESPONSE
        
            # Build record dict
            record = {
                "sentence_index": sentence_idx,
                "sentence": sentence_text,
                "prompt": prompt_text,
                "topic": result.topic,
                "explanation": result.explanation,
                "probability": result.probability
            }
        
            # Append to JSONL immediately
            self._append_record(record)
        
            # Keep in memory for final output if desired
            answers.append((sentence_idx, sentence_text, prompt_text, result))

        records = self.transform_answers(answers)
        return records
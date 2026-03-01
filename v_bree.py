import string
import random
import warnings
import json

import pandas as pd
import numpy as np

class Ensemble:
    def __init__(self, client, response_type: str, verbose: bool = False):
        self.result_cols = ["id", "iteration", "model", "question", "choices", "domain", "prompt_tokens", "previous_answer", "updated_answer", "selected_choice",
                             "response_tokens", "scoring_model", "score", "score_moving_avg", "score_moving_variance", "chosen_response", "confidence_score"]
        self.results = pd.DataFrame(columns = self.result_cols)
        self.chosen_results = pd.DataFrame(columns = self.result_cols)
        self.client = client
        self.verbose = verbose
        self.models = []
        self.variance_threshold = 9.5
        self.variance_scaling_factor = 1.1
        self.variance_confidence_coefficient = 1.0
        self.mean_confidence_coefficient = 1.0
        self.n_confidence_coefficient = 1.0
        self.response_format = None
        self.choices = []
        self.instructions = (
            "Task:\n"
            "1. Rate the answer from 0 to 100 by aggregating scores for clarity (0-30 points), completeness (0-30 points), and accuracy (0-40 points).  Empty responses should receive the minimum score.\n"
            "2. Refine the answer for maximum clarity, completeness, and accuracy. Remove filler and omit feedback or references to the original version.  Restrict your response to 600 tokens.  If no improvements are possible, provide the text verbatim."
        )
        self.mcq_instructions = "3. Based on your updated response, identify the single letter from the possible choices provided corresponding to the best answer."

        if response_type not in ["logic", "choice"]:
            raise ValueError("response_type must be either 'logic' or 'choice'")
        else:
            self.response_type = response_type
 
    def _format_choices(self, choices: list):
        cleaned_choices = [choice for choice in choices if choice is not None and choice == choice]
        choice_list = [f"{string.ascii_uppercase[i]}. {choice}" for i, choice in enumerate(cleaned_choices)]

        return choice_list

    def _build_prompt(self, user_query: str, current_text: str, choices: list = []):
        prompt = ""

        self.user_query = f"Question: {user_query}"
        self.current_answer = f"Existing Answer: {current_text}"
        if len(choices) > 0:
            choices_str = "\n".join(self._format_choices(choices))
            self.choices = f"Possible Choices:\n{choices_str}"

        ##BUILD AFTER CHOICES TO ENSURE RESPONSE FORMAT IS CORRECT
        self.build_response_format()

        if len(self.choices) == 0:
            prompt = (
                f"{self.user_query}\n"
                f"{self.current_answer}\n\n"
                f"{self.instructions}"
            )
        elif len(self.choices) > 0:
            prompt = (
                f"{self.user_query}\n"
                f"{self.current_answer}\n\n"
                f"{self.choices}\n\n"
                f"{self.instructions}\n"
                f"{self.mcq_instructions}"
            )

        return prompt

    def _extract_response(self, response: str) -> dict:
        """Extracts score and updated response from the JSON response."""
        try:
            extracted_response = json.loads(response)
            ##HANDLE CASES WHERE RESPONSE DOES NOT ADHERE TO FORMAT, ASSUMING LOW SCORE AND RETURNING RAW RESPONSE FOR REVIEW
            if not isinstance(extracted_response, dict):
                extracted_response = {"score": 0, "response": extracted_response, "letter": ""}
            return extracted_response
        except json.JSONDecodeError as e:
            extracted_response = {"score": 0, "response": "Error", "letter": ""}
            if self.verbose:
                print(f"Error decoding JSON response: {e}")
            return extracted_response
            # raise ValueError(f"Error decoding JSON response: {e}")

    ##UPDATE VARIANCE THRESHOLD TO INCREASE LIKELIHOOD OF CONVERGENCE AS MORE ROUNDS ARE COMPLETED
    def _scale_variance(self, variance_threshold: float):
        variance_threshold = variance_threshold ** self.variance_scaling_factor
        return variance_threshold

    def _best_response(self, df: pd.DataFrame):
        ##FILTER DF TO INCLUDE NON-ERROR RESPONSES ONLY - ADDED 20260228
        df = df[df["updated_answer"] != "Error"]

        sorted_results = df.sort_values(
            by=['score_moving_variance', 'score_moving_avg', 'score'], 
            ascending=[True, False, False]
        )

        best_index = sorted_results.index[0]
        
        return best_index

    def _calculate_confidence_score(self, mean, variance, n):
        parameterized_variance = variance * self.variance_confidence_coefficient
        parameterized_mean = mean * self.mean_confidence_coefficient
        parameterized_n = n * self.n_confidence_coefficient

        confidence_score = parameterized_mean / ((parameterized_variance * np.sqrt(parameterized_n)) + ((100 - parameterized_mean) * self.mean_confidence_coefficient) + 1e-6)  ##SMALL CONSTANT TO AVOID DIVISION BY 0
        return confidence_score

    ##ENSEMBLE CORE PROCESSING FUNCTION
    def run(self, data: pd.DataFrame, id_col: str, question_col: str, choices_col: str, domain_col: str, model_algorithm: str = "order_added", temperature: float = 0.0):
        ##VALIDATE CHOICES COLUMN
        if not data[choices_col].apply(lambda x: isinstance(x, list)).all():
            raise ValueError(f"All entries in the choices column '{choices_col}' must be of type list. Example: ['Rock, 'Paper', 'Scissors']")

        ##PROCESS DF ROWS
        for row in data.itertuples():
            if self.verbose:
                print(f"Processing {id_col}: {getattr(row, id_col)}")
            ##PROCESS INITIAL MODEL SELECTION
            model_index = 0
            if len(self.models) == 0:
                raise ValueError("Add at least one model to the ensemble using add_model.")
            else:
                if model_algorithm == "random_start":
                    if len(self.models) <= 1:
                        raise ValueError("At least 2 models are required for random selection.")
                    else:
                        model_index = random.randint(0, len(self.models) - 1)
                elif model_algorithm == "order_added":
                    model_index = 0

            row_results = pd.DataFrame(columns = self.result_cols)
            local_variance_threshold = self.variance_threshold
            iter_index = 0
            single_model_flag = False
            last_variance = float('inf')
            ##ITERATE WHILE VARIANCE ABOVE THRESHOLD, HARD CAPPED AT 99 AS SAFETY VALVE 
            while last_variance > local_variance_threshold and iter_index < 99 and single_model_flag == False:
                iter_result = dict.fromkeys(self.result_cols)
                iter_result["id"] = getattr(row, id_col)
                iter_result["iteration"] = iter_index
                iter_result["model"] = self.models[model_index]
                iter_result["question"] = getattr(row, question_col)
                iter_result["choices"] = getattr(row, choices_col)
                iter_result["domain"] = getattr(row, domain_col)
                iter_result["chosen_response"] = False
                if iter_index == 0:
                    iter_result["previous_answer"] = ""
                else:
                    previous_answer = row_results.loc[row_results.index[iter_index - 1], "updated_answer"]
                    ##TECHNICALLY COULD BE A SERIES, CONFIRM / CONVERT TO STRING
                    if hasattr(previous_answer, "iloc"):
                        previous_answer = previous_answer.iloc[0]
                    iter_result["previous_answer"] = "" if previous_answer is None else str(previous_answer)                    
                
                prompt = self._build_prompt(user_query = iter_result["question"], current_text = iter_result["previous_answer"], choices = iter_result["choices"])

                full_response = self.get_response(model = self.models[model_index], prompt = prompt, temperature = temperature)
                response = full_response[0]
                token_usage = full_response[1]

                ##ADD SCORE ELEMENTS TO PREVIOUS ITERATION RESULT
                if iter_index > 0:
                    row_results.loc[iter_index - 1, "scoring_model"] = self.models[model_index]
                    row_results.loc[iter_index - 1, "score"] = response["score"]
                    ##UPDATE MOVING AVERAGE AND VARIANCE ON LAST ENTRY
                    if iter_index >= len(self.models):  ##ENSURE MOVING AVERAGE AND VARIANCE ONLY CALCULATED ONCE WE HAVE SCORES FROM ALL MODELS IN THE ENSEMBLE
                        row_results.loc[len(row_results) - 1, "score_moving_avg"] = row_results.loc[(len(row_results) - len(self.models)):, "score"].mean()
                        last_variance = row_results.loc[(len(row_results) - len(self.models)):, "score"].var()
                        row_results.loc[len(row_results) - 1, "score_moving_variance"] = last_variance
                        row_results.loc[len(row_results) - 1, "confidence_score"] = self._calculate_confidence_score(row_results.loc[len(row_results) - 1, "score_moving_avg"], row_results.loc[len(row_results) - 1, "score_moving_variance"], len(row_results))

                iter_result["updated_answer"] = response["response"]
                iter_result["selected_choice"] = response["letter"]
                iter_result["prompt_tokens"] = token_usage["prompt_tokens"]
                iter_result["response_tokens"] = token_usage["response_tokens"]

                ##ADD RESULT TO DF
                row_results.loc[len(row_results)] = iter_result

                ##SCALE VARIANCE
                if (iter_index + 1) % len(self.models) == 0 and (iter_index + 1) > len(self.models):  ##SCALE VARIANCE THRESHOLD EVERY FULL ROUND THROUGH MODELS
                    local_variance_threshold = self._scale_variance(local_variance_threshold)

                ##ITERATE MODEL SELECTION
                if len(self.models) > 1:
                    model_index = (model_index + 1) % len(self.models)
                else:
                    if single_model_flag == False:
                        single_model_flag = True
                        warnings.warn(f"Ensemble operating in single model mode: {self.models[0]}, response collaboration disabled.")

                iter_index += 1

            ##FLAG THE SELECTED RESPONSE
            row_results.loc[row_results.index[self._best_response(row_results)], "chosen_response"] = True

            ##APPEND ROW RESULTS TO MAIN DF
            self.results = pd.concat([self.results, row_results], ignore_index = True)

    def add_model(self, model):
        self.models.append(model)

    def build_response_format(self):
        if self.response_type == "choice" and len(self.choices) > 0:
            self.response_format = ({"type": "json_schema",
                                     "json_schema": {
                                         "name": "ResponseExtractionLetter",
                                         "strict": True,
                                         "schema": {
                                             "type": "object",
                                             "properties": {
                                                 "score": {"type": "number"},
                                                 "response": {"type": "string"},
                                                 "letter": {"type": "string"}
                                                 },
                                             "required": ["score", "response", "letter"],
                                             "additionalProperties": False
                                             }
                                         }
                                    })
        else:
            self.response_format = ({"type": "json_schema",
                                     "json_schema": {
                                         "name": "ResponseExtractionNoLetter",
                                         "strict": True,
                                         "schema": {
                                             "type": "object",
                                             "properties": {
                                                 "score": {"type": "number"},
                                                 "response": {"type": "string"}
                                                 },
                                             "required": ["score", "response"],
                                             "additionalProperties": False
                                             }
                                         }
                                    })

    def get_response(self, model: str, prompt: str, temperature: float = 0.0) -> list:
        try:
            response = self.client.chat_completion(
                model = model,
                messages=[{"role": "user",
                            "content": prompt}],
                temperature = temperature, 
                # max_tokens = 500,
                response_format = self.response_format
            )
            return [self._extract_response(response.choices[0].message.content), {"response_tokens": response.usage.completion_tokens, "prompt_tokens": response.usage.prompt_tokens}]
        except Exception as e:
            if self.verbose:
                print(f"Error during API call: {e}")
                print(f"Prompt that caused the error: {prompt}")
            return [{"score": 0, "response": "Error", "letter": ""}, {"response_tokens": 0, "prompt_tokens": 0}]

    ##SETTERS AND GETTERS
    def set_instructions(self, instructions: str):
        self.instructions = instructions

    def set_mcq_instructions(self, mcq_instructions: str):
        self.mcq_instructions = mcq_instructions
    
    def set_variance_threshold(self, threshold: float):
        self.variance_threshold = threshold
    
    def set_variance_scaling_factor(self, scaling_factor: float):
        self.variance_scaling_factor = scaling_factor
    
    def set_variance_confidence_coefficient(self, coefficient: float):
        self.variance_confidence_coefficient = coefficient
    
    def set_mean_confidence_coefficient(self, coefficient: float):
        self.mean_confidence_coefficient = coefficient
    
    def set_n_confidence_coefficient(self, coefficient: float):
        self.n_confidence_coefficient = coefficient


    def get_instructions(self):
        return self.instructions
    
    def get_mcq_instructions(self):
        return self.mcq_instructions
       
    def get_variance_threshold(self):
        return self.variance_threshold

    def get_variance_scaling_factor(self):
        return self.variance_scaling_factor
    
    def get_results(self, selected_only: bool = False):
        if selected_only:
            return self.results[self.results["chosen_response"] == True]
        else:
            return self.results
    
    def get_confidence_coefficients(self):
        return {
            "variance_confidence_coefficient": self.variance_confidence_coefficient,
            "mean_confidence_coefficient": self.mean_confidence_coefficient,
            "n_confidence_coefficient": self.n_confidence_coefficient
        }
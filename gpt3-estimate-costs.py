"""
Estimate GPT-3 costs!
Author: Dylan
"""

import math
from RandomProSocial import RandomProSocial

class GPT3CostsCalculator:
    def __init__(self, 
             gpt3_model_version,
             max_number_of_tokens_in_response,
             best_of,
             n,
             max_allowable_cost=None):

        """
            Class to help keep track of GPT3 costs. This doesn't account for stop words and assumes the full 
            maximum number of tokens in response is used. Also, as estimate of the number of tokens in the prompts
            is taken using the "rule of thumb" on the openai site: 1 token ~ 4 charecters. 

            args
                gpt3_model_version: the name of the gpt3 model version used to calculate costs
                max_number_of_tokens_in_response: maximum number 

        """

        self.ACCEPTABLE_MODEL_VERSIONS = ["ada", "babbage", "curie", "davinci"]

        # Given in USD per 1k tokens 
        self.PRICE_PER_1K_TOKENS = {
            "ada" : 0.0008,
            "babbage" : 0.0012,
            "curie" : 0.0060,
            "davinci" : 0.0200
        }

        if gpt3_model_version not in self.ACCEPTABLE_MODEL_VERSIONS:
            raise NameError(f"GPT3 model version {gpt3_model_version} unknown. Acceptable model versions are {self.ACCEPTABLE_MODEL_VERSIONS}.")

        # constraints on inputs
        assert(best_of >= 1)
        assert(n >= 1)
        assert(max_number_of_tokens_in_response >= 1 and max_number_of_tokens_in_response <= 2048)

        # storing the cost-causing gpt3 model parameters
        self.gpt3_model_version = gpt3_model_version
        self.max_number_of_tokens_in_response = max_number_of_tokens_in_response
        self.best_of = best_of
        self.n = n

        # store cost constraints
        self.costs_incured_so_far = 0
        self.approx_total_tokens = 0

    def _calc_costs(self, total_number_of_tokens):
        """
            calculates the costs given the total number of tokens according to pricing information
            provided by openai: https://beta.openai.com/pricing
        """
        return total_number_of_tokens * self.PRICE_PER_1K_TOKENS[self.gpt3_model_version] / 1000

    def _get_num_tokens(self, sentence):
        """
            calculates number of tokens in sentence using rough guidance that 1 token is 4 charecters and takes
            cieling to be conservative
        """
        
        # using - sentence.count(' ') to omit spaces from tokens
        return math.ceil((len(sentence) - sentence.count(' ')) / 4)

    def _calc_gpt_cost(self, prompts):
        """
            calculates the costs given a set of prompts

            the formula used to perform this calculation is from: https://beta.openai.com/pricing
            see "How is pricing calculated for Completions" section
        """

        total_prompts = len(prompts)

        # tokens in responses provided by gpt3, 
        total_tokens_in_responses = total_prompts * self.max_number_of_tokens_in_response * max(self.n, self.best_of)

        # tokens in prompts provided to gpt3
        total_tokens_in_prompts = sum([self._get_num_tokens(s) for s in prompts])

        # the number of tokens overall, given settings
        total_tokens_overall = total_tokens_in_prompts + total_tokens_in_responses

        # total costs due to tokens
        total_cost = self._calc_costs(total_tokens_overall)

        return total_tokens_overall, total_cost

    def get_prompt_costs(self, prompts, store=True):
        """
            returns and stores (if store flag set) cost of prompts
            
            args
                prompts: list of prompts in form ["prompt1", "prompt2",...,"promptN"]

            returns
                cost: cost of sending all prompts to gpt3 
        """

        n_tokens, cost = self._calc_gpt_cost(prompts)

        if store:
            self.costs_incured_so_far += cost
            self.approx_total_tokens += n_tokens

        return cost

    def get_total_cost_so_far(self):
        """
            gets the total costs incurred so far
        """
        return self.costs_incured_so_far

    def __str__(self):

        string = ""

        model_info = f"{self.gpt3_model_version} gpt3 model total costs so far: "
        total_costs = f"~${round(self.get_total_cost_so_far(),2)} USD, from ~{self.approx_total_tokens} tokens."

        string += model_info
        string += total_costs

        return string

## Example usage

p1 = ["going to the store", "testing prompts", "woah this prompt is super cool"]
p2 = ["blah blah blah blah"] * 100
p3 = ["testing one two three four"] * 10_000

calculator = GPT3CostsCalculator("davinci", max_number_of_tokens_in_response=5, best_of=1, n=1)


# total = 0
# for _ in range(10):
#     r = RandomProSocial('prosocial_dialog_v1/test.json')
#     running_cost = 0 
#     for i in range(500):
#         running_cost += calculator.get_prompt_costs(r.get_prompt()[0])
#     total+=running_cost
# print(total / 10)

print(calculator._get_num_tokens("__possibly_needs_intervention__"))
#####


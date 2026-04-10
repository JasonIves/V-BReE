Adapted from the [V-BReE Template Colab Notebook](https://colab.research.google.com/drive/1bqVaN_cPq14h1aeevEvmCxSMgNBz9udM)

Tested using [MMLU-PRO](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)

---

# Documentation and workspace setup for the V-BReE framework.

## 1. Adding the V-BReE framework to a workspace

### Import the V-BReE module from Github
To install the V-BReE module, clone the git repository into the colab file system.

`from pathlib import Path`

`vbree_path = Path("v_bree.py")`

`if not vbree_path.is_file():`

  `!wget https://raw.githubusercontent.com/JasonIves/V-BReE/main/v_bree.py`

---

*If you run into problems with folder naming, you should be able to delete or rename the existing folder and then create a fresh clone*

### Import the module

You can now import the *v_bree* module.

`import v_bree`

## 2. Configuring the V-BReE framework workspace

### Store inference connection API key
Modeling and inference connections often require an API key.  That key can be stored in the "Secrets" section of the sidebar, given a name, and then referenced using the *userdata* library.  This example represents a HuggingFace Inference Client key, but other services should work similarly.

Do not store API keys directly in code.

Once the token is stored you will need to give the notebook permission to access it.

### Define inference connection client

Define a client connection that uses the OpenAI API format.  Primary testing for this module was done using the [Hugging Face Inference Client](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client), but other clients utilizing the OpenAI API format should work as well.

## 3. Working with the V-BReE framework

### General system configuration

As usual, finishing outfitting the workspace with the necessary Python tools.

### Load data

Load the desired dataset, either from local storage, Google Drive, or a 3rd party download.  For primary development and testing [MMLU-PRO](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) was downloaded from Hugging Face, and converted to a Pandas dataframe.

### Configure the data for V-BReE compatibility

A V-BReE compatibile dataset has 4 key components.  Each must be provided, although the domain can be set to a dummy value without impacting ensemble processing.

- An identifier that is unique for each row. Ex: "*12893*"
- The question being asked of the ensemble.  This should be a string. Ex: "*What is 2 + 2?*"
- The possible choices, formatted as a list. Ex: "*[3, 4, 5]*"
  - Actual choices are not required. But an empty list, Ex: "*[]*", should still be submitted in the designated choices column.  When detected the ensemble will proceed in free-response mode instead of MCQ mode.
- Question domain.  Not required. For conveyence to results data. Ex: "*math*"

Any other data submitted to the V-BReE ensemble will be ignored.

### Instantiate Ensemble class

Create an instance of the Ensemble class, this will be the workflow manager for the V-BReE framework.

Parameters:
- *client* - Object. API inference client
- *response_type* - String, *"logic"* or *"choice"*.  Define the type of response you want from the models.  *"logic"* will return logic only, *"choice"* will return both choice and logic.
- *verbose* - Boolean, default *False*. Control display of status messages.

`e = v_bree.Ensemble(client = client, response_type = "choice", verbose = True)`

### Check for available models

Identify models for ensembling.  For testing and development HfApi with a "text-generation" filter was used to identify the constituent models.

Commonly chosen HfApi authors are:
- openai
- google
- meta-llama
- Qwen
- mistralai
- deepseek-ai

### Add models to Ensemble

Add a string refrence for each model you want to inclue in the ensemble.  The format of these strings may vary depending on the client, and you may need to pre-load the models to your workspace if you are using them locally.

Pre-loading is not necessary for Hugging Face Inference Client.

There is no set limit to the number of models that can be added.  Prompting a single model configuration will return non-ensemble, single-prompt results.

`e.add_model("openai/gpt-oss-20b:groq")`

`e.add_model("Qwen/Qwen2.5-7B-Instruct:together")`

`e.add_model("meta-llama/Llama-3.1-8B-Instruct:cerebras")`

### Run the V-BReE ensemble

To run the ensemble, pass:
- *data* - DataFrame. Properly formatted data frame.
- *id_col* - String. Column name of the unique identifier column.
- *question_col* - String. Column name of the question text column.
- *choices_col* - String. Column name of the list-formatted choices column.
- *domain_col* - String. Column name of the question domain column.
- *model_algorithm* - String, default *"order_added"*, *"order_added"* or *"random_start"*. Flag for whether the ensemble should always start with the first model added, or start with a random model.
- *temperature* - Float, default *0.0*. Desired temperature that the models should process at.

`e.run(data = sample,`
      `id_col = "question_id",`
      `question_col = "question",`
      `choices_col = "options",`
      `domain_col = "category",`
      `model_algorithm = "random_start",`
      `temperature = 0.0)`

## Review the ensemble results

Retrieve the results from the ensemble.
- *selected_only* - Boolean. Flag indicating if you want the selected responses returned, or all responses processed by the ensemble.

`results = e.get_results(selected_only = True)`

`display(results)`

### Process the results as needed

Once the results are returned you can write to csv, check correctness of responses, generate plots, summary statistics, etc. - as with any other data.

## 4. Other Available V-BReE Methods

A variety of other methods for getting and setting various ensemble parameters are also available.

#### Setters:
*set_instructions(instructions: str)* - Set primary instructional prompt text.

*set_mcq_instructions(mcq_instructions: str)* - Set MCQ specific instructional prompt text.

*set_variance_threshold(threshold: float)* - Set the variance threshold starting value.

*set_variance_scaling_factor(scaling_factor: float)* - Set the variance threshold exponential scaling factor.

*set_variance_confidence_coefficient(coefficient: float)* - Set the variance confidence coefficient, for parameterizing the variance in the confidence formula.
    
*set_mean_confidence_coefficient(coefficient: float)* - Set the mean confidence coefficient, for parameterizing the mean in the confidence formula.
    
*set_n_confidence_coefficient(coefficient: float)* - Set the response count confidence coefficient, for parameterizing the response count in the confidence formula.


#### Getters:
*get_instructions()* - Get the current primary instructional prompt text.
    
*get_mcq_instructions()* - Get the current mcq specific instructional prompt text.
       
*get_variance_threshold()* - Get the current variance threshold starting value.

*get_variance_scaling_factor()* - Get the current variance scaling factor.
    
*get_confidence_coefficients()* - Get a dict of variance, mean and response count confidence coefficients.

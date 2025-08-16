

import os
import re
import ast
import json
import random
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

# If you're using AzureOpenAI specifically
from openai import AzureOpenAI  
# Or switch to OpenAI if that's what you're using:
from openai import OpenAI

prompt = """
You are an expert in environmental psychology. Your task is to classify the environmental attitude expressed in a given statement.

For each statement, return:

1. **Main Category** (choose one from the taxonomy below)
2. **Subcategory** (select the best-fitting one from within that category)
3. **Justification** (1-2 sentences explaining your reasoning)

---

### TAXONOMY

**Nature & Human Relationship**
- Biocentrism/Ecocentrism: Nature has intrinsic value; all species have equal rights.
- Anthropocentrism: Human needs come first; nature is a resource.
- Spiritual Stewardship: Nature is sacred; humans have a religious/moral duty to protect it.

**Technology & Development**
- Techno-Optimism: Technology can solve environmental problems.
- Techno-Skepticism: Technology causes harm or alienation.
- Urbanization/Infrastructure: Views on housing, transport, or land development.

**Policy & Governance**
- Regulatory Support: Pro-environmental laws, taxes, or rules.
- Government Skepticism: Opposition to environmental regulations.
- Internationalism: Global treaties, cooperation, or UN involvement.
- Environmental Justice: Fairness, marginalized communities, or equitable impacts.

**Personal Identity & Responsibility**
- Environmental Identity: Sees self as an environmentalist.
- Personal Behavior: Mentions recycling, conserving, or consumer habits.
- Activism & Engagement: Protests, donations, organizing, or civic action.

**Environmental Threat Perception**
- Climate Concern: Belief in climate urgency or risk.
- Climate Denial/Skepticism: Minimizes environmental threats.
- Pollution Concern: Air, water, or industrial pollution.
- Resource Scarcity: Concern about depletion of water, land, energy, etc.

**Consumption & Materialism**
- Eco-Consumerism: Favors eco-friendly or green products.
- Anti-Consumerism: Critiques materialism; supports simplicity.

**Intergenerational & Future Orientation**
- Future Generations: Motivated by the wellbeing of future descendants.
- Sustainability Ethics: Supports long-term ecological balance over short-term gains.

---

### EXAMPLES

**Statement**: "We must protect forests, even if it means sacrificing economic development."  
→ Main Category: Nature & Human Relationship  
→ Subcategory: Biocentrism/Ecocentrism  
→ Justification: The speaker values forests independently of human benefit, implying intrinsic worth of nature.

---

**Statement**: "Technology will help us solve climate change faster than any policy ever could."  
→ Main Category: Technology & Development  
→ Subcategory: Techno-Optimism  
→ Justification: The statement expresses strong faith in technological solutions to environmental problems.

---

**Statement**: "I always switch off the lights when I leave the room because I care about energy conservation."  
→ Main Category: Personal Identity & Responsibility  
→ Subcategory: Personal Behavior  
→ Justification: The speaker describes an individual action that reflects environmental consciousness.

---

### NOW CLASSIFY:

**Statement**: <SEED>

"""

prompt_fem = """
You are an expert in gender studies and political psychology. Your task is to classify the attitude toward feminism and gender equality expressed in a given statement.

For each statement, return:

1. **Main Category** (choose one from the taxonomy below)  
2. **Subcategory** (select the best-fitting one from within that category)  
3. **Justification** (1-2 sentences explaining your reasoning)

---

### TAXONOMY

**1. Workplace & Economic Equality**
- Pay Equity: Belief in equal pay for equal work or critiques of wage gaps.
- Workplace Discrimination: Mentions bias, barriers to promotion, or unfair treatment at work.
- Affirmative Action / Quotas: Support or opposition to reparative policies.
- Work-Life Balance & Childcare: Views on maternity leave, daycare, or role strain.

**2. Gender Roles & Domesticity**
- Traditional Roles: Belief that women's primary duty is homemaking and caregiving.
- Egalitarian Roles: Support for equal division of labor and decision-making in families.
- Parenting & Marriage Norms: Attitudes about who should have authority or caregiving responsibility.
- Role Reversal Acceptance: Comfort with men in caregiving or women as providers.

**3. Feminist Identity & Movement Engagement**
- Self-Identification as Feminist: Expresses personal alignment with feminism.
- Movement Participation: Protests, petitions, donations, or activism for women's rights.
- Feminist Solidarity: Feels affinity with feminist individuals or groups.
- Disidentification / Ambivalence: Rejects or distances self from feminism.

**4. Political & Civic Representation**
- Support for Representation: Advocates for more women in leadership, politics, or public roles.
- Opposition to Representation: Asserts men are better suited for leadership or politics.

**5. Structural & Systemic Critique**
- Patriarchy Critique: Identifies male dominance as a root cause of women's inequality.
- Intersectionality: Mentions race, class, or sexuality in relation to gender issues.
- Capitalism & Gender: Connects women's oppression to economic systems.
- Institutional Reform: Proposes changes to laws, media, religion, or education.

**6. Reproductive Rights & Sexual Autonomy**
- Abortion Rights: Support for or opposition to a woman's right to choose.
- Sexual Norms & Double Standards: Addresses gendered expectations around sex or courtship.
- Autonomy in Relationships: Advocates for women's independence in romantic or sexual decisions.

**7. Gender Stereotypes & Cultural Norms**
- Media & Beauty Standards: Concerns about portrayal of women in culture and media.
- Language & Respectability Norms: Views on how women should speak or act.
- Essentialist Beliefs: Claims about “natural” differences between men and women.

**8. Anti-Feminist / Backlash Attitudes**
- Dismissal of Inequality Claims: Argues that women are not oppressed.
- Feminism Gone Too Far: Believes feminist demands are excessive or unnecessary.
- Equality Already Achieved: Asserts that gender parity already exists.

---

### EXAMPLES

**Statement**: "Women and men should be paid equally for the same work."  
→ Main Category: Workplace & Economic Equality  
→ Subcategory: Pay Equity  
→ Justification: This expresses clear support for wage equality between genders.

**Statement**: "A woman's main role is to be a good wife and mother."  
→ Main Category: Gender Roles & Domesticity  
→ Subcategory: Traditional Roles  
→ Justification: The speaker affirms a traditional view of a woman's domestic responsibilities.

**Statement**: "I attend rallies supporting women's rights whenever possible."  
→ Main Category: Feminist Identity & Movement Engagement  
→ Subcategory: Movement Participation  
→ Justification: The speaker describes civic activism on behalf of feminist causes.

---

### NOW CLASSIFY:

**Statement**: <SEED>

"""



####################################################################################
def generate_ai_response(prompt, temperature, model_type=None):
    """
    Generates an AI response using the Azure OpenAI API based on the specified system role description.

    Args:
    - role_description (str): Description of the AI's role.
    - model_name (str): Name or identifier of the model to use.
    - max_tokens (int): Maximum number of tokens to generate.
    - temperature (float): Controls randomness in the generation.
    - top_p (float): Nucleus sampling parameter.
    - frequency_penalty (float): Decreases the likelihood of repeatedly used words.
    - presence_penalty (float): Decreases the likelihood of repeatedly used topics.
    - stop_sequence (str or None): Sequence where the generation will stop.

    Returns:
    - str: The generated text response from the AI.
    """
    # Initialize the Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint="https://zitopenai.openai.azure.com/",
        api_key="",
        api_version="2024-02-15-preview"
    )

    llamaclient = OpenAI(
        base_url = "https://chat-ai.academiccloud.de/v1",
        api_key = "c038da40bbde8910104048c4b149129c",

    )


    # Define the message text for the system
    message_text = [
        {"role": "system", "content": prompt}
    ]

    if model_type == "open-source":
        print("Open source activated!")
        client = llamaclient
        model = "meta-llama-3.1-70b-instruct" # Choose any available model
        #model = "o1"
        # model = "meta-llama-3.1-8b-instruct",
        # - meta-llama-3.1-8b-instruct
        # - mistral-large-instruct
        # - meta-llama-3.1-70b-instruct
        # - qwen2.5-72b-instruct      
        print("Open source activated!!!!  ", model)  
    else:
        model = "gpt-4o"

    # Create a chat completion request with the specified parameters
    completion = client.chat.completions.create(
        model=model,
        messages=message_text,
        temperature=temperature,
        #max_tokens=,
        seed=89
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop_sequence
    )

    # Return the AI's generated response
    try:
        response = completion.to_dict()['choices'][0]['message']['content'].strip()
        return response
    except KeyError as exc:
        print(exc)
        print("ERROR: ")
        print(prompt)
        print("---"*20)
        print(completion.to_dict())
        #raise KeyError(exc)


def load_responses_to_dataframe(folder, num_files):
    responses = []
    for i in range(num_files):
        # Load the response
        response = joblib.load(f'{folder}/response_{i}.joblib')
        
        # Append the response to the list
        responses.append(response)
    
    # Create a DataFrame from the list of responses
    responses_df = pd.DataFrame(responses, columns=['Response'])
    return responses_df

def save_to_joblib(response, folder, index, prefix):
    """
    Save the response to a joblib file in the specified folder.
    """
    try:
        joblib.dump(response, f'{folder}/{prefix}_{index}.joblib')
    except FileNotFoundError:
        os.mkdir(folder)
        joblib.dump(response, f'{folder}/{prefix}_{index}.joblib')

# Traverse up until the root of the filesystem
while os.getcwd() != '/':
    if os.path.exists('.git'):
        break
    os.chdir('..')





def process_batch(df, start_index, batch_size, folder, prompt_template, temperature, mapping_classes):
    """
    Processes a batch of rows from the DataFrame.

    Args:
        df (pd.DataFrame): The dataset to process.
        start_index (int): The starting index for the batch.
        batch_size (int): The size of the batch.
        folder (str): The folder to save the joblib files.
        prompt_template (str): The prompt template.
    """
    end_index = start_index + batch_size
    batch_df = df.iloc[start_index:end_index]

    for _, row in batch_df.iterrows():
        prompt = prompt_template.replace(
            "<SEED>",
            f"{row['Item content']}"
        )

        response = generate_ai_response(prompt, temperature=temperature, model_type='open-source')

        print("----"*50)
        print(row['Item content'], response)
        print("----"*50)
        
        save_to_joblib(response, folder, row.name, 'response')
        #save_to_joblib(mapping_classes, folder, row['gid'], 'mapping_classes')



if __name__ == "__main__":
    # Load the dataset
    print(os.listdir())
    # df = pd.read_csv('notebooks/revision_sistematica__environmentalism__clustered_v1.csv')
    df = pd.read_csv('notebooks/revision_sistematica__feminism__clustered_v1.csv')
    # Output folder for batch processing
    # batch_id = '20250321_t70_flat__subset_2__gpt4-o'
    batch_id = '20250713_feminism_llama_v1'

    try:
        n_processed = len(os.listdir(batch_id))
    except FileNotFoundError:
        os.mkdir(batch_id)
        n_processed = 0

    # Define batch size
    batch_size = 50  # Number of rows to process per call

    # Total rows in the dataset
    
    total_rows = len(df)

    # Process in batches until all rows are processed
    while n_processed < total_rows:

        #classification_schema_mapped = {i: class_ for i, class_ in enumerate(classification_schema_.keys())}
        # classes = list(class_definitions.keys())

        process_batch(
            df=df,
            start_index=n_processed,
            batch_size=batch_size,
            folder=batch_id,
            prompt_template=prompt_fem,
            temperature=0.7,
            mapping_classes = None
        )
        n_processed += batch_size